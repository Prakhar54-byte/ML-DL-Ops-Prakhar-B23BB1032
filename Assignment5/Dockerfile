FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN echo "[Docker build] Step 1/8: apt update/install" && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN echo "[Docker build] Step 2/8: set python alternative" && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN echo "[Docker build] Step 3/8: upgrade pip" && \
    python -m pip install --upgrade pip

WORKDIR /work

COPY requirements.txt /work/requirements.txt
RUN echo "[Docker build] Step 4/8: install requirements" && \
    pip install -r /work/requirements.txt

ENV PYTHONUNBUFFERED=1
