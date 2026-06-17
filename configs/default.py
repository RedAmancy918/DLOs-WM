"""默认配置。改这里调实验。"""

DEFAULT_CONFIG = {
    # 数据 / 物理
    "num_nodes": 24,
    "contact_radius": 0.04,
    "traj_len": 20,
    "traj_per_epoch": 16,

    # 模型
    "hidden": 128,
    "n_message_passing": 6,
    "n_topo_classes": 3,
    "dt": 0.04,

    # 训练
    "lr": 1e-3,
    "epochs": 30,
    "noise_std": 0.002,          # >0 开启 noise injection（pushforward 近似）
    "pushforward_weight": 0.5,

    # 多 head 权重
    "weights": {
        "pos": 1.0,
        "tension": 0.5,
        "contact": 0.3,
        "topo": 0.3,
        "fail": 0.3,
    },

    # failure label 派生规则
    "tension_limit": 8.0,         # 张力超过此值算 failure（按你的物理标定）
    "stuck_topo_classes": [2],    # 这些拓扑类视为"卡死"
}
