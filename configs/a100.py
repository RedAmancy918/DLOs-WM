"""A100(80GB) 压测配置。覆盖 default 里的小规模设置。

这些是"先吃满单卡再说"的起点值，压测脚本会帮你逐档加压找上限。
"""

A100_CONFIG = {
    "num_nodes": 64,          # 绳子离散点数（真实任务可到 64~128）
    "hidden": 256,
    "n_message_passing": 10,
    "batch_size": 256,        # 每卡每 step 的图数（disjoint 拼一起）
    "traj_per_epoch": 128,
    "traj_len": 24,
    "epochs": 50,
    "lr": 3e-4,
}
