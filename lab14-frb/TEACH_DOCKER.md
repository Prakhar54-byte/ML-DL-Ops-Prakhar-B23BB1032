# 🐳 Docker for Deep Learning: The definitive Guide

A complete guide to setting up reproducible, GPU-accelerated environments for Deep Learning (DL) using Docker and Docker Compose.

## 📖 Table of Contents

1. [Why Docker?](https://www.google.com/search?q=%23why-docker)
2. [Prerequisites & Architecture](https://www.google.com/search?q=%23prerequisites--architecture)
3. [Part 1: The Dockerfile](https://www.google.com/search?q=%23part-1-the-dockerfile)
4. [Part 2: Building & Running (CLI)](https://www.google.com/search?q=%23part-2-building--running-cli)
5. [Part 3: Docker Compose (The Better Way)](https://www.google.com/search?q=%23part-3-docker-compose-the-better-way)
6. [Common Pitfalls & Fixes](https://www.google.com/search?q=%23common-pitfalls--fixes)

---

## Why Docker?

In Deep Learning, dependency hell is real. You might need CUDA 11.8 for one project and CUDA 12.1 for another. Docker encapsulates your entire environment (OS, Drivers, Python, PyTorch/TensorFlow) into an isolated container.

* **Reproducibility:** Code runs exactly the same on your laptop, a university cluster, or AWS.
* **Isolation:** No more conflicts between system-wide Python libraries.
* **Portability:** Share a single file (`Dockerfile`) instead of a 10-page installation wiki.

---

## Prerequisites & Architecture

To run deep learning containers, your host machine needs three things:

1. **Docker Engine:** The core software to run containers.
2. **NVIDIA Drivers:** Installed on the *host* machine (OS level). The container shares these drivers.
3. **NVIDIA Container Toolkit:** The bridge that allows Docker to "see" your GPU.

### 🛠️ Setup Check

Before proceeding, verify the toolkit is working:

```bash
# This should print your GPU details from inside a test container
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

```

*If this fails, install the `nvidia-container-toolkit` for your OS.*

---

## Part 1: The Dockerfile

The `Dockerfile` is your blueprint. It tells Docker how to build your environment image.

**File:** `Dockerfile`

```dockerfile
# 1. Base Image: Use an official NVIDIA image with CUDA support
# Always match the CUDA version to the PyTorch/TF version you intend to use.
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# 2. Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# 3. Setup workspace
WORKDIR /workspace

# 4. Install essential system tools
# 'git' for cloning, 'htop' for monitoring, 'libgl1' for OpenCV
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    vim \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy requirements and install Python dependencies
# We copy requirements.txt FIRST to leverage Docker caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. (Optional) Create a non-root user for security
# This prevents permission issues with created files on Linux
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# 7. Default command
CMD ["python", "train.py"]

```

---

## Part 2: Building & Running (CLI)

### 1. Build the Image

```bash
docker build -t dl-project:v1 .

```

### 2. Run the Container (The Modern Way)

We use the `--gpus` flag, which is the standard for Docker 19.03+.

```bash
docker run -it --rm \
  --gpus all \
  --shm-size=8g \
  -v $(pwd):/workspace \
  dl-project:v1

```

**Breakdown of Flags:**

* `--gpus all`: **Crucial.** Passes all GPUs to the container. You can also use `--gpus '"device=0,1"'` to select specific GPUs.
* `--shm-size=8g`: **Crucial for PyTorch.** Increases shared memory. The default (64MB) is too small for DataLoaders and will cause "Bus Error".
* `-v $(pwd):/workspace`: **Volume Mounting.** Maps your current folder (host) to `/workspace` (container). Changes made to code inside the container are reflected instantly on your host.
* `-it`: Interactive mode (so you can see logs and use Ctrl+C).
* `--rm`: Removes the container automatically when you exit.

---

## Part 3: Docker Compose (The Better Way)

Typing long Docker commands is error-prone. **Docker Compose** allows you to define your run configuration in a YAML file. This is the industry standard for development.

Create a file named `docker-compose.yml`:

```yaml
services:
  training:
    # Build from the Dockerfile in the current directory
    build: 
      context: .
      dockerfile: Dockerfile
    
    # Image name to tag
    image: dl-project:latest
    
    # Enable GPU support (The modern 'deploy' specification)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all # Or set to 1, 2, etc.
              capabilities: [gpu]
    
    # Shared memory size (Prevents DataLoader crashes)
    shm_size: '8gb'
    
    # Volumes: Map your code and datasets
    volumes:
      - .:/workspace              # Sync current code
      - ./data:/workspace/data    # Mount dataset folder
      - ./logs:/workspace/logs    # Persist logs/checkpoints
    
    # Keep container running (useful for debugging/Jupyter)
    command: tail -f /dev/null
    
    # Environment variables
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=your_key_here

```

### How to use Compose:

1. **Start the environment:**
```bash
docker compose up -d --build

```


*(The `-d` runs it in the background)*
2. **Enter the container:**
```bash
docker compose exec training bash

```


*Now you are inside the container with GPUs enabled!*
3. **Run your script:**
```bash
python train.py

```


4. **Stop everything:**
```bash
docker compose down

```



---

## Common Pitfalls & Fixes

### 1. `RuntimeError: DataLoader worker (pid) is killed by signal: Bus error`

* **Cause:** Docker's default shared memory is too low for PyTorch.
* **Fix:** Add `--shm-size=8g` (CLI) or `shm_size: '8gb'` (Compose).

### 2. Permission Denied on generated files

* **Cause:** Docker runs as root by default, so files it creates (like `model.pth`) are owned by root.
* **Fix:** Run the container with your user ID:
```bash
docker run --user $(id -u):$(id -g) ...

```



### 3. "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver"

* **Cause:** Mismatch between Host driver version and Container CUDA version, or Toolkit not installed.
* **Fix:** Ensure your Host NVIDIA drivers are up to date. You do **not** need to install CUDA Toolkit on the host, only the Drivers. The Container provides the CUDA Toolkit.
