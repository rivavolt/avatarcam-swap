"""avatarcam swap server: replaces the face in a client's camera frames with a chosen face, using FaceFusion's HyperSwap 1a 256 with the occlusion mask.

Runs inside the RunPod pod from the avatarcam-swap image (FaceFusion, onnxruntime-gpu with its CUDA/cuDNN wheels and the models baked in), started by bootstrap.sh. One TCP client at a time; every message in both directions is `type:u8 | len:u32be | payload`.

client -> server
  A  auth token (utf-8), must come first
  P  optional json options line, then the face image bytes (png/jpeg); answered with R once the face is detected and the swapper is warm. A later P swaps the face mid-stream.
  F  seq:u32be | jpeg camera frame

server -> client
  R  json {gpu, load_s, prepare_s, since_boot_s, workers}
  F  seq:u32be | server_us:u32be | jpeg frame, the same size as the camera frame
  E  error text, then the server closes the connection

Frames run on WORKERS threads at once, which is what holds 30 fps when one frame takes longer than a frame period, and leave in the order they arrived. At most MAX_QUEUE frames wait for a worker; beyond that the oldest waiting frame is dropped, so latency stays bounded rather than growing. A frame where no face is found would show the real face, so the last swapped frame is held for HOLD frames and after that the frame goes out pixelated.

The server writes a heartbeat file on every received frame; bootstrap.sh's watchdog deletes the pod when it goes stale, and this process also deletes the pod itself when no frame has arrived for AVATARCAM_IDLE seconds.

`python server.py --prefetch` downloads the models the server uses, which is how the image bakes them in.
"""

import collections
import concurrent.futures
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

FF = os.environ.get("AVATARCAM_FF", "/opt/facefusion")
PORT = int(os.environ.get("AVATARCAM_PORT", "8765"))
IDLE = float(os.environ.get("AVATARCAM_IDLE", "300"))
HEARTBEAT = os.environ.get("AVATARCAM_HEARTBEAT", "/tmp/avatarcam-heartbeat")
WARM_FACE = os.environ.get("AVATARCAM_WARM_FACE", "/opt/avatarcam/face.jpg")
JPEG_Q = int(os.environ.get("AVATARCAM_JPEG_Q", "85"))
WORKERS = int(os.environ.get("AVATARCAM_WORKERS", "4"))
MAX_QUEUE = 2
HOLD = int(os.environ.get("AVATARCAM_HOLD", "15"))
FF_ARGS = [
    "--processors", "face_swapper",
    "--face-swapper-model", "hyperswap_1a_256",
    "--face-mask-types", "box", "occlusion",
    "--face-selector-mode", "one",
    "--execution-providers", "cuda",
    "--video-memory-strategy", "tolerant",
    "--log-level", "info",
]

last_activity = time.time()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def touch():
    global last_activity
    last_activity = time.time()
    try:
        with open(HEARTBEAT, "w") as f:
            f.write(str(int(last_activity)))
    except OSError:
        pass


def terminate_pod(reason):
    pod = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    log("terminating pod", pod, "reason:", reason)
    # the pod-scoped key is refused by the REST API but may terminate its own pod through GraphQL
    if pod and key:
        q = json.dumps({"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}).encode()
        req = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data=q,
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json", "user-agent": "avatarcam"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                log("podTerminate:", r.status, r.read()[:200])
        except Exception as e:
            log("podTerminate failed:", e)
    # bootstrap.sh retries the termination when this process is gone
    os._exit(3)


def idle_watchdog():
    while True:
        time.sleep(10)
        if time.time() - last_activity > IDLE:
            terminate_pod(f"no client frames for {IDLE:.0f}s")


def facefusion(face_path):
    """Imports FaceFusion from its checkout and applies the server's options, the same way its own CLI does."""
    os.chdir(FF)
    sys.path.insert(0, FF)
    import cv2
    import onnxruntime

    # A pod gets a CPU quota far below the host's core count, while onnxruntime and OpenCV size their thread pools by the host; that many spinning threads under the quota throttle every frame, so each session gets a small non-spinning pool.
    cv2.setNumThreads(2)
    session_cls = onnxruntime.InferenceSession

    def session(path, providers=None, **kw):
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        return session_cls(path, sess_options=so, providers=providers, **kw)

    from facefusion import inference_manager, logger, program, state_manager
    from facefusion.args import apply_args
    from facefusion.program import create_program

    inference_manager.InferenceSession = session
    # the parser lists ffmpeg's encoders for the video output options, which the stream path never uses, and the image has no ffmpeg
    program.get_available_encoder_set = lambda: {"audio": [], "video": []}
    sys.argv = ["facefusion.py", "headless-run", "--source-paths", face_path, "--target-path", face_path, "--output-path", "/tmp/avatarcam-unused.jpg", *FF_ARGS]
    apply_args(vars(create_program().parse_args()), state_manager.init_item)
    logger.init("info")
    return state_manager


def prefetch():
    facefusion(WARM_FACE)
    from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer
    from facefusion.processors.modules.face_swapper import core as face_swapper

    # the content analyser and the voice extractor are common modules of every processor but nothing on the stream path runs them
    face_swapper.get_common_modules = lambda: [face_classifier, face_detector, face_landmarker, face_masker, face_recognizer]
    if not face_swapper.pre_check():
        sys.exit("model download failed")
    log("models present")


class Engine:
    """FaceFusion's stream path (the function its own webcam mode runs per frame) with the current face."""

    def __init__(self):
        t0 = time.time()
        self.state = facefusion(WARM_FACE)
        import cv2
        import numpy as np
        from facefusion.processors.modules.face_swapper import core as face_swapper
        from facefusion.streamer import process_stream_frame
        from facefusion.vision import read_static_images

        self.cv2, self.np = cv2, np
        self.face_swapper = face_swapper
        self.process_stream_frame = process_stream_frame
        self.read_static_images = read_static_images
        self.lock = threading.Lock()
        self.faces = 0
        self.set_face_path(WARM_FACE)
        # The threads that swap a client's frames. They live as long as the engine because onnxruntime's CUDA provider sets up per-thread state (stream, cuBLAS and cuDNN handles) on a thread's first run of each model, which costs seconds, so these threads are the ones warmed here.
        self.pool = concurrent.futures.ThreadPoolExecutor(WORKERS, thread_name_prefix="swap")
        # every model's session is created on its first use; doing that from several threads at once creates some twice, so the first frames go one at a time
        warm = np.full((720, 1280, 3), 200, np.uint8)
        face = cv2.imread(WARM_FACE)
        face = cv2.resize(face, (480, 480))
        warm[120:600, 400:880] = face
        for _ in range(3):
            self.process_stream_frame(self.source_frames, warm)
        # a barrier holds each task until all WORKERS have one, so every pool thread runs the models at least once
        for _ in range(2):
            barrier = threading.Barrier(WORKERS)

            def run():
                barrier.wait(timeout=120)
                self.process_stream_frame(self.source_frames, warm)

            list(self.pool.map(lambda _: run(), range(WORKERS)))
        self.load_s = time.time() - t0

    def set_face_path(self, path):
        self.state.set_item("source_paths", [path])
        if not self.face_swapper.pre_process("stream"):
            raise ValueError("no face found in the face image")
        self.source_frames = self.read_static_images([path])

    def set_face(self, image_bytes):
        t0 = time.time()
        img = self.cv2.imdecode(self.np.frombuffer(image_bytes, self.np.uint8), self.cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("the face image does not decode")
        # a fresh path per face, since FaceFusion caches images and faces by path
        with self.lock:
            self.faces += 1
            path = f"/tmp/avatarcam-face-{self.faces}.png"
        self.cv2.imwrite(path, img)
        self.set_face_path(path)
        # the first frame with a new face computes its embedding
        warm = self.np.full((720, 1280, 3), 200, self.np.uint8)
        warm[120:600, 400:880] = self.cv2.resize(img, (480, 480))
        self.process_stream_frame(self.source_frames, warm)
        return time.time() - t0

    def swap(self, bgr):
        """The swapped frame, or None when no face was found in it."""
        out = self.process_stream_frame(self.source_frames, bgr)
        return None if self.np.array_equal(out, bgr) else out

    def pixelate(self, bgr):
        h, w = bgr.shape[:2]
        small = self.cv2.resize(bgr, (max(1, w // 24), max(1, h // 24)), interpolation=self.cv2.INTER_AREA)
        return self.cv2.resize(small, (w, h), interpolation=self.cv2.INTER_LINEAR)


def gpu_name():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip().splitlines()[0]
    except Exception:
        return "?"


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("client closed")
        got += k
    return bytes(buf)


def recv_msg(sock):
    t, n = struct.unpack(">BI", recv_exact(sock, 5))
    if n > 32 << 20:
        raise ValueError(f"message too large: {n}")
    return chr(t), recv_exact(sock, n)


def pack(t, payload):
    return struct.pack(">BI", ord(t), len(payload)) + payload


def serve_client(conn, engine_box, engine_ready, token):
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    t, payload = recv_msg(conn)
    if t != "A" or payload != token:
        conn.sendall(pack("E", b"bad token"))
        raise PermissionError("bad token")
    log("client authenticated")
    touch()

    cond = threading.Condition()
    pending = collections.deque()
    results = {}
    stop = threading.Event()
    counters = {"next_ticket": 0, "next_send": 0, "dropped": 0, "missed": 0, "n": 0, "busy": 0.0}
    held = {"frame": None, "misses": 0}
    send_lock = threading.Lock()

    def send(msg):
        with send_lock:
            conn.sendall(msg)

    def fail(text):
        try:
            send(pack("E", text.encode()))
        except OSError:
            pass
        stop.set()
        with cond:
            cond.notify_all()

    def worker(engine):
        cv2, np = engine.cv2, engine.np
        while not stop.is_set():
            with cond:
                while not pending and not stop.is_set():
                    cond.wait(0.5)
                if stop.is_set():
                    return
                seq, jpg = pending.popleft()
                ticket = counters["next_ticket"]
                counters["next_ticket"] += 1
            try:
                t0 = time.perf_counter()
                bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                out = engine.swap(bgr) if bgr is not None else None
                with cond:
                    if out is not None:
                        held["frame"], held["misses"] = out, 0
                    else:
                        counters["missed"] += 1
                        held["misses"] += 1
                        if bgr is not None and (held["frame"] is None or held["misses"] > HOLD or held["frame"].shape != bgr.shape):
                            out = engine.pixelate(bgr)
                        else:
                            out = held["frame"]
                msg = None
                if out is not None:
                    data = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()
                    us = int((time.perf_counter() - t0) * 1e6)
                    msg = pack("F", struct.pack(">II", seq, us) + data)
            except Exception as e:
                traceback.print_exc()
                fail(f"frame failed: {e}")
                return
            with cond:
                # a frame that does not decode still takes its ticket, so the sender does not wait for it forever
                results[ticket] = msg
                if msg is not None:
                    counters["n"] += 1
                    counters["busy"] += us
                    if counters["n"] % 300 == 0:
                        log(f"frames={counters['n']} mean_server_ms={counters['busy'] / counters['n'] / 1000:.1f} dropped={counters['dropped']} no_face={counters['missed']}")
                cond.notify_all()

    def sender():
        while not stop.is_set():
            with cond:
                while counters["next_send"] not in results and not stop.is_set():
                    cond.wait(0.5)
                if stop.is_set():
                    return
                msg = results.pop(counters["next_send"])
                counters["next_send"] += 1
            if msg is None:
                continue
            try:
                send(msg)
            except OSError:
                stop.set()
                return

    def prepare(face, opts):
        engine_ready.wait()
        engine = engine_box[0]
        if isinstance(engine, Exception):
            fail(f"engine failed to load: {engine}")
            return
        try:
            prep = engine.set_face(face)
        except Exception as e:
            traceback.print_exc()
            fail(str(e))
            return
        info = {
            "gpu": gpu_name(),
            "load_s": round(engine.load_s, 2),
            "prepare_s": round(prep, 2),
            "since_boot_s": round(time.time() - float(os.environ.get("AVATARCAM_BOOT_T0", time.time())), 1),
            "workers": WORKERS,
        }
        log("ready", info)
        if not workers:
            workers.extend(engine.pool.submit(worker, engine) for _ in range(WORKERS))
        try:
            send(pack("R", json.dumps(info).encode()))
        except OSError:
            stop.set()

    workers = []
    sender_thread = threading.Thread(target=sender, daemon=True)
    sender_thread.start()
    try:
        while not stop.is_set():
            t, payload = recv_msg(conn)
            touch()
            if t == "F":
                (seq,) = struct.unpack(">I", payload[:4])
                with cond:
                    pending.append((seq, payload[4:]))
                    while len(pending) > MAX_QUEUE:
                        pending.popleft()
                        counters["dropped"] += 1
                    cond.notify_all()
            elif t == "P":
                opts = {}
                img = payload
                if payload[:1] == b"{":
                    nl = payload.index(b"\n")
                    opts = json.loads(payload[:nl])
                    img = payload[nl + 1 :]
                # off the receive loop, so a face swap mid-stream does not stall the frames
                threading.Thread(target=prepare, args=(img, opts), daemon=True).start()
    finally:
        stop.set()
        with cond:
            cond.notify_all()
        sender_thread.join(timeout=3)
        concurrent.futures.wait(workers, timeout=3)
        log(f"client done: frames={counters['n']} dropped={counters['dropped']} no_face={counters['missed']}")


def main():
    if "--prefetch" in sys.argv:
        prefetch()
        return
    token = os.environ["AVATARCAM_TOKEN"].encode()
    touch()
    threading.Thread(target=idle_watchdog, daemon=True).start()
    engine_box = [None]
    engine_ready = threading.Event()

    def load():
        try:
            engine_box[0] = Engine()
            log(f"engine warm in {engine_box[0].load_s:.1f}s")
        except Exception as e:
            traceback.print_exc()
            engine_box[0] = e
        engine_ready.set()

    threading.Thread(target=load, daemon=True).start()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(4)
    log(f"listening on :{PORT}, idle limit {IDLE:.0f}s")
    while True:
        conn, addr = srv.accept()
        log("connection from", addr)
        try:
            serve_client(conn, engine_box, engine_ready, token)
        except Exception as e:
            log("client ended:", repr(e))
        finally:
            conn.close()


if __name__ == "__main__":
    main()
