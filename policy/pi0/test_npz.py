import numpy as np
import torch

# 1️⃣ 加载 npz 文件
npz_path = "/mnt/data/weiyang.hu/RoboTwin-main/policy/pi0/pi0_params.npz"
data = np.load(npz_path)

print(f"Loaded {len(data.files)} parameters from {npz_path}\n")

# 2️⃣ 遍历打印每层信息
for k in data.files:
    v = data[k]
    print(f"{k}: shape={v.shape}, dtype={v.dtype}")

# 3️⃣ 可选：转成 PyTorch tensor dict
torch_params = {k: torch.from_numpy(data[k]) for k in data.files}

# # 4️⃣ 打印一层示例
# example_key = list(torch_params.keys())[0]
# print(f"\nExample tensor for '{example_key}':\n", torch_params[example_key])
