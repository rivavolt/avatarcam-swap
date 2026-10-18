# avatarcam's swap image, ghcr.io/rivavolt/avatarcam-swap: FaceFusion 3.9.1, onnxruntime-gpu with the CUDA 12 and cuDNN 9 wheels it loads, and the models server.py uses, so a pod serves without downloading anything. The pod's code (bootstrap.sh, server.py) arrives in its environment from the client; the copy of server.py here only names the models to bake.
#
# Built by the rivavolt/avatarcam-swap repository's workflow from a mirror of this directory (publish-image.sh). The heavy wheels go in separate layers so a pull fetches them in parallel.
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git libgomp1 && rm -rf /var/lib/apt/lists/*

ARG FF_TAG=3.9.1
RUN git clone -q --depth 1 --branch ${FF_TAG} https://github.com/facefusion/facefusion /opt/facefusion && rm -rf /opt/facefusion/.git

RUN python -m venv /opt/ff
RUN /opt/ff/bin/pip install -q "nvidia-cudnn-cu12~=9.0"
RUN /opt/ff/bin/pip install -q "nvidia-cublas-cu12~=12.0"
RUN grep -v -E '^(onnxruntime|gradio)' /opt/facefusion/requirements.txt > /tmp/req.txt \
 && /opt/ff/bin/pip install -q -r /tmp/req.txt "onnxruntime-gpu[cuda,cudnn]==1.24.4" \
 && rm /tmp/req.txt

COPY face.jpg server.py /opt/avatarcam/
RUN /opt/ff/bin/python /opt/avatarcam/server.py --prefetch && du -sh /opt/facefusion/.assets/models && ls /opt/facefusion/.assets/models

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
