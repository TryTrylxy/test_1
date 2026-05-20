import os

# 尝试使用 GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

print("🔍 程序2：尝试使用 GPU 0...")

try:
    # 尝试在 GPU 上创建张量
    x = torch.randn(1000, 1000, device="cuda")
    print("❌ 失败：GPU 0 没有被独占，仍然可用")
except RuntimeError as e:
    print("=====================================")
    print("✅ 成功！GPU 0 已被独占！")
    print("✅ 其他程序无法使用这张显卡")
    print(f"错误信息：{e}")
    print("=====================================")