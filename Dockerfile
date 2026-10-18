# avatarcam's swap image, ghcr.io/rivavolt/avatarcam-swap: FaceFusion 3.9.1, onnxruntime-gpu with the CUDA 12 and cuDNN 9 wheels it loads, and the models server.py uses, so a pod serves without downloading anything. The pod's code (bootstrap.sh, server.py) arrives in its environment from the client; the copy of server.py here only names the models to bake.
#
# Built by the rivavolt/avatarcam-swap repository's workflow from a mirror of this directory (publish-image.sh). The image is about 3 GB compressed and its pull is most of a pod's cold start, so the final stage lays it out as several layers of a few hundred MB each, which a pull fetches in parallel.
FROM python:3.12-slim-bookworm AS build

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git libgomp1 && rm -rf /var/lib/apt/lists/*

ARG FF_TAG=3.9.1
RUN git clone -q --depth 1 --branch ${FF_TAG} https://github.com/facefusion/facefusion /opt/facefusion && rm -rf /opt/facefusion/.git

RUN python -m venv /opt/ff \
 && grep -v -E '^(onnxruntime|gradio)' /opt/facefusion/requirements.txt > /tmp/req.txt \
 && /opt/ff/bin/pip install -q -r /tmp/req.txt "onnxruntime-gpu[cuda,cudnn]==1.24.4"

COPY face.jpg server.py /opt/avatarcam/
RUN /opt/ff/bin/python /opt/avatarcam/server.py --prefetch && du -sh /opt/facefusion/.assets/models

# /stage/<n> holds layer n's files at their final paths; whatever no earlier group claims lands in the last one
SHELL ["/bin/bash", "-c"]
RUN set -e; SP=/opt/ff/lib/python3.12/site-packages; M=/opt/facefusion/.assets/models; n=0; \
    group() { n=$((n + 1)); for p in "$@"; do [ -e "$p" ] || continue; mkdir -p "/stage/$n$(dirname "$p")"; mv "$p" "/stage/$n$p"; done; }; \
    group $SP/nvidia/cudnn/lib/libcudnn_engines_precompiled*; \
    group $SP/nvidia/cudnn; \
    group $(ls -S $SP/nvidia/cublas/lib/* | head -1); \
    group $SP/nvidia/cublas; \
    group $SP/nvidia; \
    group $SP/onnxruntime; \
    group $M/hyperswap_1a_256.onnx; \
    group $M; \
    group /opt; \
    du -sh /stage/*

FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /stage/1/ /
COPY --from=build /stage/2/ /
COPY --from=build /stage/3/ /
COPY --from=build /stage/4/ /
COPY --from=build /stage/5/ /
COPY --from=build /stage/6/ /
COPY --from=build /stage/7/ /
COPY --from=build /stage/8/ /
COPY --from=build /stage/9/ /
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
