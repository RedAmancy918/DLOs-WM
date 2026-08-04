# DLO World Model (GNN)

一个基于图神经网络的 **可变形线状物体（绳/线/线缆）世界模型** 脚手架。
不预测图像，而是直接预测结构化物理量，作为"学习型物理模拟器"，
用于预测动作后果，支撑 cable routing / insertion / knot tying / unknotting 等长程任务。

纯 PyTorch 手写 message passing，**零额外依赖**（除 torch 本身）。

## 它预测什么（多 head）

| head | 含义 | 监督 |
|---|---|---|
| `pos_next` / `acc` | 下一帧 centerline 几何 | MSE |
| `tension` | 沿弧长的张力场 | Huber（抗长尾）|
| `contact` | 每节点接触概率 | BCE |
| `topology` | 图级拓扑类别 | CE |
| `failure` | 失败风险 | BCE，**label 由 tension/topology 派生**，不是独立信号 |

## 架构

GNS / DPI-Net 一脉的 **encode–process–decode**：
- 节点 = centerline 离散点；结构边 =(i,i+1) 弹性段；接触边 = 动态自接触
- Processor：6 轮 message passing（边更新 → 聚合 → 节点更新，带残差）
- 动作（双臂抓取-位移）散射成每节点驱动信号喂入每一轮
- 多 head decoder：节点级（pos/tension/contact）+ 图级 pooling（topology/failure）

## 运行

```bash
pip install torch
python scripts/run.py
```

会用内置 `SyntheticRope` 玩具数据训练 + 做 rollout 评估，
保存 `runs/model.pt` 与 `runs/report.json`。

> 注意：`SyntheticRope` 只是自洽的玩具动力学，用来打通管线、对齐维度、抓 bug。
> **不是真实物理，不可当真。**

## DLO-Lab 与 world model 的分工

DLO-Lab 能运行 20 步甚至更长的物理仿真；它在本项目中充当数据生成器和
ground-truth evaluator，而不是要学习的预测模型。真正的研究目标是一个结构化
DLO world model：给定初始状态、episode 级材料条件和未来动作序列，递归预测
未来的 centerline、速度、张力、接触与拓扑状态。

| 组件 | 输入 | 输出 | 角色 |
|---|---|---|---|
| DLO-Lab | 初始物理状态、材料、动作 | 物理仿真轨迹 | 生成训练/验证/测试真值 |
| 材料条件化 GNN | `state_t`、材料、`action_t` | 预测 `state_{t+1}` | 学习单步动力学 |
| GNN closed-loop rollout | `state_0`、材料、动作序列 | 递归预测 `state_1...state_T` | 20 步 world-model 评测 |

这里的 horizon 按 **macro step** 计数。DLO-Lab 内部物理步长是 1 ms，
`steps_interval` 表示一个 world-model step 包含多少个物理子步。例如：

- `steps_interval=200`：一个 macro step 为 0.2 s；
- `horizon=20`：20 个动作、21 帧状态，共覆盖 4 s；
- 训练可使用较短的 truncated rollout loss（例如 10 步），测试仍可从模型自己的
  预测状态递归到 20 步。

因此，DLO-Lab 的 20 步轨迹是监督真值；论文要验证的是学习型 world model 能否
在不读取未来真值的情况下稳定预测同一段 20 步轨迹。仓库此前使用的 2-step
`/tmp` 数据只是端到端 smoke test，不是正式长时结果。

## Workshop：材料条件化闭环管线

当前 v2 管线把 `K/E/G`、线密度、半径和自摩擦作为 episode-level
`MaterialCondition`，支持 DLO-Lab 数据生成、材料条件化模型、同容量零条件
baseline、旧 GNN baseline，以及 ID/OOD/paired-counterfactual 评测。

`DLOState` 只保存真正的动态状态。自接触边和节点 contact 会在 rollout 每一步按
当前预测位置重新派生；SDF 尚未纳入本阶段模型，也不会作为可递推状态缓存。

```bash
# 20-step 管线示例；正式 episode 数量和 OOD/CF 切分见 workshop_milestone.md
python scripts/gen_material_dataset.py \
  --out-dir runs/workshop_data \
  --horizon 20 --steps-interval 200
python scripts/run_material.py \
  --data runs/workshop_data/train.pt \
  --rollout-horizon 10 --val-horizon 20 \
  --out runs/workshop_material.pt
python scripts/eval_material.py \
  --checkpoint runs/workshop_material.pt \
  --id-data runs/workshop_data/test_id.pt \
  --ood-data runs/workshop_data/test_ood_material.pt \
  --counterfactual-data runs/workshop_data/counterfactual.pt \
  --horizon 20
```

训练和评测入口会校验同目录 `manifest.json` 的 SHA256、生成源码/动作协议契约、
`macro_dt` 和接触构图规则；材料 shuffle 的实际置换也会写入评测 JSON。完整的
训练入口默认读取 `train.pt` 同目录的 `val.pt`，按闭环 position NRMSE 选回最佳
epoch，并以 patience 10 early-stop。冻结协议、必跑消融和 go/no-go 标准见
[`docs/workshop_milestone.md`](docs/workshop_milestone.md)。

## 接入你自己的仿真数据（核心）

只需实现一个 `TrajectoryProvider`（见 `dlo_wm/data/dataset.py`）：

```python
from dlo_wm.data.schema import DLOState, DLOAction
from dlo_wm.data.dataset import TrajectoryProvider

class MyIsaacRope(TrajectoryProvider):
    @property
    def num_nodes(self): return 24
    def sample_trajectory(self, T=20):
        # 从你的 DER / XPBD / Isaac / MuJoCo 仿真里取一条轨迹，
        # 整理成 schema 约定的 DLOState 序列：
        #   pos[N,3] vel[N,3] tension[N] contact[N] topology(scalar)
        # 以及 DLOAction 序列、每帧 contact_pairs[K,2]
        return states, actions, contact_pairs
```

然后在 `scripts/run.py` 里把 `provider = SyntheticRope(...)`
换成 `provider = MyIsaacRope(...)`，其余不动。

**真实数据要点**（来自设计讨论）：
- tension / contact force 的 ground truth 只能从仿真拿；真实数据靠多视角重建 centerline + 力传感器锚点
- topology label 用 Gauss linking / crossing number / knot invariant 自动算，替换玩具版 `_topology_label`
- 物理参数（弯曲/扭转刚度、摩擦、线密度）应作为条件特征喂入并做 domain randomization
- 长程任务（knot tying）需 scripted policy / demonstration 提供成功骨架 + 周围扰动，random rollout 采不到成功轨迹

## 已知硬骨头（值得先小实验验证）
- 长 horizon rollout 误差累积 —— 已内置 noise injection（pushforward 近似）缓解，可在 config 调
- 拓扑变化瞬间（穿越/打结）边的重连 —— 当前靠几何距离推断接触边，复杂拓扑建议混合符号化/几何判定模块兜底

## 目录
```
dlo_wm/
  data/schema.py       状态 / 动作 / 材料 schema 与派生图特征
  data/dataset.py      TrajectoryProvider 接口 + 玩具生成器
  data/provenance.py   数据 SHA256 与生成协议契约
  model/gnn.py         GNN、材料条件编码与闭环 rollout
  train/losses.py    多 head 损失 + failure label 派生
  train/material_trainer.py  单步 + differentiable closed-loop 训练
  eval/material_rollout.py   ID/OOD/paired-CF 指标
scripts/gen_material_dataset.py  DLO-Lab v2 数据生成
scripts/run_material.py          workshop 训练入口
scripts/eval_material.py         workshop 评测入口
```
