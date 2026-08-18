# RoboTwin pi0 Docker 构建指南

单一 `Dockerfile`，通过 `--build-arg` 切换 CUDA 变体：

| 构建命令 | 镜像标签 | torch 来源 | 说明 |
|----------|----------|------------|------|
| `docker build -t robotwin-pi0:cu124 .` | `robotwin-pi0:cu124` | uv.lock（PyPI，自带 cu124 nvidia 库） | 与宿主机编译的 curobo `.so` 直接匹配 |
| `docker build --build-arg ... -t robotwin-pi0:cu128 .`（见下文） | `robotwin-pi0:cu128` | 显式重装 `download.pytorch.org/whl/cu128` | 与基础镜像 CUDA 12.8.1 原生匹配 |

基础镜像：cu124 默认 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`；cu128 用 `--build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04`。
预期体积：约 30GB（不含 `assets`）。

> **Python 版本警示**：venv 必须使用 uv 托管的 standalone CPython（≥3.11.13）。
> Ubuntu 22.04 apt 仓库的 `python3.11` 是 **3.11.0rc1 预发布版**，其 PEP 659
> 自适应特化有 bug（`_Py_Specialize_StoreAttr` 对 torch `ScriptFunction` 实例
> 存 `__doc__` 时 `dk=NULL` 解引用），curobo 导入期 `torch.jit.script` 必段错误。
> 因此 cu128 构建以 `uv python install 3.11` 创建 venv（与宿主机一致）。

## 适用环境

本仓库 **不提交** curobo `.so` 等大文件。目标使用者应已在本机完成：

1. `policy/pi0` 的 `uv sync`（venv 已就绪）
2. `envs/curobo` 的编译安装（`curobolib/*.so` 已生成）

## 前置条件

- Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- 磁盘空间 ≥ 60GB
- 可访问 `ghcr.io`（拉取 uv 基础层）

## 快速开始

```bash
cd /path/to/RoboTwin

# clone lerobot + 检查 venv / curobo .so
bash script/prepare-docker-build.sh

# 选其一构建：
# cu124（默认 ARG）：
docker build -t robotwin-pi0:cu124 .
# cu128（覆盖 ARG）：
docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
  --build-arg UV_VERSION=0.9.18 \
  --build-arg CUDA_VARIANT=cu128 \
  --build-arg EXTRA_LD_LIB_PATH=/usr/local/cuda/lib64: \
  -t robotwin-pi0:cu128 .
```

构建后验证（以 cu128 为例，cu124 同理替换标签）：

```bash
docker run --rm --gpus all robotwin-pi0:cu128 bash -lc '
  source /workspace/RoboTwin/policy/pi0/.venv/bin/activate
  python -c "import torch; print(\"cuda\", torch.cuda.is_available(), torch.version.cuda)"
  python -c "from curobo.wrap.reacher.motion_gen import MotionGen; print(\"curobo ok\")"
  python /workspace/RoboTwin/script/test_render.py
'
```

预期：`cuda True`、`curobo ok`、`Render Well`。

## Git 提交策略（本分支）

| 内容 | 是否提交 git | 来源 |
|------|-------------|------|
| `Dockerfile`、`.dockerignore`、文档、脚本 | ✅ 提交 | 仓库 |
| `policy/pi0/vendor/lerobot` | ❌ 不提交 | `prepare-docker-build.sh` build 前 clone |
| `envs/curobo/`（含 `.so`） | ❌ 不提交 | 本地 curobo build |
| `docker-vendor/` | ❌ 不提交 | 已废弃，无需维护 |
| `assets/`、`eval_data/` | ❌ 不提交 | 挂载或本地准备 |

## 本地准备 curobo .so

若尚未编译 curobo：

```bash
cd policy/pi0 && source .venv/bin/activate
cd ../../envs/curobo && pip install -e . --no-build-isolation
```

完成后应存在：

```
envs/curobo/src/curobo/curobolib/*.so   # 5 个文件
```

Docker build 时 `COPY envs/curobo` 会带上这些 `.so`，**无需** `docker-vendor/`。

`.so` 须与容器内 Python 3.11 / CUDA / torch 版本匹配。注意 cu128 镜像的 torch 来自 `download.pytorch.org/whl/cu128`（当前为 2.11.0+cu128，非 uv.lock 的 2.6.0），因此编译 `.so` 的宿主机 venv torch 必须与之一致（宿主机 venv 同为 2.11.0+cu128 时可直接复用）。

## lerobot（build 前自动 clone）

```bash
bash script/prepare-docker-build.sh
# 或手动：
git clone https://github.com/huggingface/lerobot policy/pi0/vendor/lerobot
cd policy/pi0/vendor/lerobot && git checkout a445d9c9da6bea99a8972daa4fe1fdd053d711d2
```

网络不稳定时可用浅拉取单 commit（数据量小）：

```bash
mkdir -p policy/pi0/vendor/lerobot && cd policy/pi0/vendor/lerobot
git init
git remote add origin https://github.com/huggingface/lerobot
git fetch --depth 1 origin a445d9c9da6bea99a8972daa4fe1fdd053d711d2
git checkout FETCH_HEAD
```

## 运行时挂载

| 路径 | 说明 |
|------|------|
| `assets/` | 仿真资源（约 16GB），必须 `-v` 挂载 |
| `eval_result/` | 评测输出，建议挂载 |
| PyTorch 模型 | 按需挂载到 `policy/pi0/torch_model/...` |

## 导出与分发

```bash
docker save robotwin-pi0:cu128 -o robotwin-pi0-cu128.tar
docker load -i robotwin-pi0-cu128.tar
# cu124 同理：
docker save robotwin-pi0:cu124 -o robotwin-pi0-cu124.tar
```

## 常见问题

**Q: 缺少 curobo .so 导致 build 失败？**  
先在本机 `pip install -e .` 编译 curobo，再运行 `bash script/prepare-docker-build.sh`。

**Q: cu124 和 cu128 该选哪个？**  
- 宿主机 curobo `.so` 是在 cu124 torch 下编译的 → 选 cu124（直接匹配，无需重装 torch）。
- 想用与基础镜像 CUDA 12.8 原生匹配的 torch → 选 cu128（构建时重装 torch）。

**Q: 还需要 docker-vendor 吗？**  
不需要。旧方案已移除，可直接删除本地 `docker-vendor/` 目录。

**Q: build 从 `COPY . .` 开始重编？**  
改业务代码会 bust 该层，但 `uv sync` / curobo 等前置层仍 CACHED。
