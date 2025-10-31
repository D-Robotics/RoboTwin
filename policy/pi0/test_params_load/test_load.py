import torch
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

# 加载预训练模型
policy = PI0Policy.from_pretrained("lerobot/pi0")

# 示例 batch，需包含至少一个图像和任务指令
batch = {
    "image": torch.randn(1, 3, 224, 224),      # 这里仅为示例，替换为实际图像
    "goal": ["do something"]                  # 示例目标指令
}

# 执行动作选择
action = policy.select_action(batch)

print("Predicted action:", action)
