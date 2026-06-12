FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 系统依赖：curobo 编译 + sapien/open3d 渲染
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs curl ca-certificates \
    build-essential cmake ninja-build \
    libegl1-mesa-dev libgl1-mesa-glx libglew-dev libglfw3-dev \
    libgles2-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    libosmesa6-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:0.9.1 /uv /uvx /bin/

WORKDIR /workspace/RoboTwin

# 复制 uv sync 构建 openpi 所需的全部文件（hatchling 需要 LICENSE）
COPY policy/pi0/pyproject.toml policy/pi0/uv.lock policy/pi0/.python-version policy/pi0/LICENSE policy/pi0/
COPY policy/pi0/packages policy/pi0/packages
COPY policy/pi0/src policy/pi0/src
COPY policy/pi0/vendor/lerobot policy/pi0/vendor/lerobot

# 用 apt 装 Python，避免 uv 从 GitHub 下载超时
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建 venv（UV_* 放这里，避免改动时冲掉 apt 缓存）
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never
RUN cd policy/pi0 && \
    sed -i 's|lerobot = { git = "https://github.com/huggingface/lerobot", rev = "a445d9c9da6bea99a8972daa4fe1fdd053d711d2" }|lerobot = { path = "vendor/lerobot" }|' pyproject.toml && \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --python python3.11 && \
    cp -r src/openpi/models_pytorch/transformers_replace/* \
       .venv/lib/python3.11/site-packages/transformers/

# curobo：使用宿主机已编译 .so（需先完成 curobo build，见 scripts/prepare-docker-build.sh）
COPY envs/curobo envs/curobo
RUN test -n "$(ls envs/curobo/src/curobo/curobolib/*.so 2>/dev/null)" || \
    (echo "ERROR: 缺少 curobo .so，请先编译 curobo 或运行 bash scripts/prepare-docker-build.sh" && exit 1)
RUN cd policy/pi0 && . .venv/bin/activate && \
    cd ../../envs/curobo && \
    sed -i 's/ext_modules=ext_modules/ext_modules=[]/' setup.py && \
    sed -i 's/cmdclass={"build_ext": BuildExtension}/cmdclass={}/' setup.py && \
    pip install -e . --no-build-isolation && \
    pip install warp-lang==1.8.0

# 运行时 apt（置于 COPY . . 之前，改代码或尾部层时不重跑 apt）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvulkan1 vim \
    && rm -rf /var/lib/apt/lists/*

# 运行时环境变量（稳定，置于 COPY . . 之前）
ENV VIRTUAL_ENV=/workspace/RoboTwin/policy/pi0/.venv \
    PATH="/workspace/RoboTwin/policy/pi0/.venv/bin:${PATH}" \
    PYTHONPATH=/workspace/RoboTwin:/workspace/RoboTwin/policy/pi0/src \
    NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/workspace/RoboTwin/policy/pi0/.venv/lib/python3.11/site-packages/torch/lib \
    CUROBO_TORCH_COMPILE_DISABLE=0 \
    VK_ICD_FILENAMES=/workspace/RoboTwin/policy/pi0/.venv/lib/python3.11/site-packages/sapien/vulkan_library/nvidia_icd.json

# 再复制完整代码（排除项见 .dockerignore）
COPY . .

# 本地化补丁（curobo .git 被 **/.git 排除，需单独恢复）
COPY envs/curobo/.git envs/curobo/.git
RUN sed -i 's|path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})|path = Path("/workspace/RoboTwin/policy/pi0/src/openpi/shared/big_vision/paligemma_tokenizer.model")|' \
    policy/pi0/src/openpi/models/tokenizer.py


WORKDIR /workspace/RoboTwin
CMD ["/bin/bash"]
