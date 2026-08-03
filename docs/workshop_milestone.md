# NeurIPS Workshop 阶段实验协议（Workshop v1）

> 目标：用最小但闭环完整的证据回答“显式物理条件是否改善 DLO 世界模型的长时预测与材料反事实预测”。本文是预注册式执行协议，不是结果报告；任何数值结果都必须由固定数据与 3 个训练 seed 实际跑出后再填写。

## 0. 当前状态与边界

状态标记：`[已有]` 表示代码已出现在当前工作树；`[待验证]` 表示仍需在 DLO-Lab 环境跑测试；`[待实现]` 表示本协议要求但当前尚无完整入口；`[待运行]` 表示不能提前写结论。

| 项目 | 状态 | 说明 |
|---|---|---|
| v2 `MaterialCondition` / `DLOEpisode` 与序列化 | [已有] | 材料是 episode 级条件，动态状态仍只含 `pos/vel/tension/contact/topology`；单元测试已过 |
| episode 级 K/E/线密度/半径随机化 | [已有] | G 与自摩擦在 Workshop v1 中固定，零方差维不会产生伪 OOD 响应 |
| DLO-Lab setter 与 paired counterfactual 生成 | [已有][待完整数据 QA] | solver 读回冒烟已过；反事实组共享控制 seed、绝对目标和未沉降初态 |
| 单图/批图材料条件化模型 | [已有] | 材料编码进入输入与消息传递层；Workshop CLI 当前采用单图训练路径 |
| 材料归一化、闭环训练与 rollout 指标 | [已有][待规模化运行] | train-only normalizer、val NRMSE 选模/early-stop、`macro_dt`、动态边/contact 和 raw 指标均显式校验 |
| 三个复现脚本 | [已有][待规模化运行] | 远端最小生成→三模型训练→ID/OOD/CF 评测均已跑通 |
| 数据与结果 provenance | [已有] | split/manifest/checkpoint SHA256、生成源码指纹、动作协议契约与 shuffle 映射均落盘并校验 |
| 论文表格、图和统计结论 | [待运行] | 当前不得声称任何方法优于 baseline |

本阶段只做均匀材料、固定节点数、固定初始自然几何和已存在的 `random/fold/loop` 操作。真实机器人、视觉重建、异质材料、未知参数在线辨识和新任务规划均不属于最小投稿闭环。

## 1. 研究问题与最小 claim

### 1.1 主问题

在相同 DLO 几何状态和抓手控制下，把已知材料参数作为显式 episode 条件送入 GNN，是否能：

1. 降低 ID 与材料 OOD 的 20 步闭环 rollout 误差；
2. 正确预测“只改变一个材料量”造成的轨迹差异；
3. 在自接触和拓扑变化发生时保持稳定，而不只改善单步 teacher-forced loss。

### 1.2 最小可投稿 claim

只有满足第 10 节 go 条件后，摘要才可使用以下表述：

> 在 DLO-Lab 的受控均匀材料设置中，显式 episode 级物理条件相对同架构无条件模型，改善材料外推的长时 centerline rollout，并更准确地复现 paired material intervention 的轨迹效应。

claim 必须限定为模拟器、已知材料条件、固定几何离散和本协议参数范围。不能外推到真实世界泛化、任意障碍物、任意绳结或材料参数辨识。

### 1.3 本阶段不把 SDF 作为核心 claim

- `obstacle_sdf`、`obstacle_normal`、`nearest_obstacle_id` 不进入 `DLOState`，也不写入 v2 episode 的逐帧持久状态。
- Workshop v1 不用 SDF 作为模型输入，不把障碍物泛化写进标题、摘要或主表。
- rollout 中结构边和自接触边必须由当前预测 `pos` 每步重建，禁止缓存第 0 步的派生边。
- 如果后续增加障碍物条件，环境几何只保存在 episode metadata；SDF 必须由 `sdf_builder(pred_pos, obstacle_metadata)` 每步派生，与动态 edge builder 对称。该扩展不阻塞本阶段投稿。

## 2. 固定实验单位与数据契约

- 节点数 `N=64`，轨迹长度 `T=20`，默认 `steps_interval=200`、仿真 `dt=1e-3 s`，因此宏步长 `0.2 s`、总预测时长 `4.0 s`。
- 每条 episode 保存 `T+1` 个 state、`T` 个 action、`T+1` 组 contact pairs、一个不随时间变化的 `MaterialCondition`。
- 数据先按完整 episode 分 split，再展开 transition；禁止随机拆 transition。
- 所有材料特征 normalizer 只在 train episode 上拟合，并随 checkpoint 保存。
- train/val/test/OOD/counterfactual 使用不重叠的 episode seed 范围；manifest 保存生成参数、git commit、DLO-Lab commit、格式版本和文件哈希。
- 模型选择只看 validation 的 `position_NRMSE@20`；ID test、OOD 和 counterfactual 在配置冻结前不可查看。
- `random/fold/loop` 在普通 split 中各占三分之一；若数量不能整除，按 manifest 记录的 `--motions` 顺序轮转，多出的样本依次给列表前项。

## 3. 材料定义、ID 与 OOD

基准材料使用当前 provider 默认值：

- `K0=5e4`，`E0=1e5`，`G0=1e4`；
- `rho0 = N * node_mass0 / sum(rest_length)`，不把顶点质量错误地除以逐段自然长度；
- `r0=0.005`，`mu_static=0.30`，`mu_kinetic=0.25`。

Workshop v1 只随机化 `K/E/rho/r`。`G`、静摩擦和动摩擦仍记录并送入统一 schema，但保持常数，因此不得声称模型已学会扭转或摩擦条件效应。

### 3.1 ID 分布

| 参数 | 采样范围（相对基准） | 分布 |
|---|---:|---|
| K | `[0.70, 1.40] * K0` | log-uniform |
| E | `[0.70, 1.40] * E0` | log-uniform |
| 线密度 rho | `[0.80, 1.20] * rho0` | uniform |
| 半径 r | `[0.90, 1.10] * r0` | uniform |
| G / 自摩擦 | `1.0 * base` | fixed |

四个 ID 变量独立采样。实际 solver setter 值和最终 `MaterialCondition` 都必须落盘，不能只保存采样 seed。

### 3.2 单轴材料 OOD

OOD 每次只把一个目标参数放到 ID 支撑集之外，其余三个变量仍从 ID 范围采样。这样能判断是哪一个物理轴失效，而不是只报告一个混合 OOD 平均数。

| 目标参数 | low-OOD | high-OOD | 每个 tail 数量 |
|---|---:|---:|---:|
| K | `[0.40, 0.60] * K0` | `[1.60, 2.00] * K0` | 75 |
| E | `[0.40, 0.60] * E0` | `[1.60, 2.00] * E0` | 75 |
| rho | `[0.60, 0.75] * rho0` | `[1.30, 1.50] * rho0` | 75 |
| r | `[0.75, 0.85] * r0` | `[1.15, 1.25] * r0` | 75 |

共 `4 parameters * 2 tails * 75 = 600` 条 OOD episode。主表必须逐轴报告，再给等权 macro average；不能让样本更多或更容易的 slice 主导平均值。

半径会影响物理接触半径和动态接触边构造。所有方法都使用当前 episode 的真实半径构图；材料 shuffle 消融只打乱 encoder 输入、保持真实构图不变。因此，只有 radius slice 改善不能单独证明显式材料 encoder 有效。

## 4. 固定 split 与数量

| split | episode 数 | transition 数 | seed 区间 | 用途 |
|---|---:|---:|---:|---|
| train-ID | 1200 | 24000 | `0..1199` | 参数拟合 |
| val-ID | 150 | 3000 | `1000000..1000149` | checkpoint / 一次性超参选择 |
| test-ID | 300 | 6000 | `2000000..2000299` | 最终 ID 报告 |
| test-OOD | 600 | 12000 | 每个 tail `3000000..3000074` | 8 个单轴 tail |
| paired-CF | 100 groups / 500 episodes | 10000 | 每个参数 `4000000..4000024` | 材料 intervention |

普通 split 总计 2250 条 episode；paired-CF 另计 500 条。paired group 是不可拆分单位，全部仅用于最终评估，不能混入 train/val。

在生成主数据前先跑 24 条 smoke episode（每种 motion 8 条）；smoke 数据只用于接口和数值范围检查，不进入任何论文结果。

## 5. Paired counterfactual 设计

对 `K/E/rho/r` 各生成 25 个 group，共 100 组。每组固定：

- 同一个 canonical `state_0`，`settle_steps=0`；
- 同一个 `control_seed`、`grasp_idx`、绝对 `target_pos` 和动作时序；
- 同一个任务，其中每个参数 13 组 `loop`、12 组 `fold`；
- 只改变指定材料参数，scale 为 `[0.50, 0.70, 1.00, 1.40, 1.80]`。

每组生成后必须自动断言：

1. 五条 episode 的初始 `pos/vel` 在容差 `1e-7` 内一致；
2. 所有 action 的抓点、激活掩码、目标位置和 duration 一致；
3. metadata 除 episode id、variant scale 和被干预参数外一致；
4. 非干预材料量逐值相等；
5. group id 唯一，整组只出现在 counterfactual test。

反事实主比较以 `scale=1.0` 为 reference。模型和 simulator 都计算最终 centerline effect：

`effect(q) = pos_T(q) - pos_T(q=1.0)`。

动态 contact edge 仍由各自 rollout 的当前预测位置和该 episode 的真实半径逐步重建；不能使用 ground-truth contact pairs 驱动模型 rollout。

## 6. 模型、baselines 与 ablations

### 6.1 必跑方法

| ID | 方法 | 目的 |
|---|---|---|
| B0 | constant-velocity rollout | 无学习 sanity baseline；抓手 drive 的处理必须在附录写清 |
| B1 | 原无条件 DLO GNN | 与 full model 使用相同 state/action、动态构图、训练数据和 processor 深度 |
| B2 | 材料条件 GNN，单步 teacher forcing | 隔离显式材料条件的贡献 |
| B3 | B2 + 真闭环多步 loss | full model；验证长时训练是否进一步改善 rollout |

B1 与 B2/B3 报告可训练参数量。若条件分支使参数量增加超过 B1 的 5%，再加一个 width-matched B1；否则不额外扩大实验矩阵。

训练协议固定为最多 100 epoch，validation `position_NRMSE@20` early-stop patience 10。只允许先用 model seed 0 和 val split 做一次超参选择；冻结配置后再从头运行全部 3 seeds。

### 6.2 必跑消融

- A1：对 B3 跨 episode 随机打乱材料 encoder 输入，但保持该 episode 的真实半径构图，检验网络是否真的使用条件。
- A2：材料只拼到输入层，不注入每轮 message passing，对比“input-only”与“every-layer conditioning”。
- A3：B2 对比 B3，报告 `@1` 与 `@20`，防止用单步收益冒充闭环收益。

有余力再做，不阻塞首轮 go/no-go：leave-one-material-feature-out、structural-edge-only、不同训练集规模曲线。禁止为了凑表临时加入没有对应研究问题的复杂 baseline。

## 7. 指标与统计

### 7.1 普通 rollout 主指标

在 horizon `1/5/10/20` 报告：

- `position_NRMSE@k = position_RMSE@k / total_rest_length`（主指标）；
- velocity RMSE、对称 Chamfer / rope length；
- tension MAE；
- 平均相对 edge-length violation；
- rollout divergence rate：非有限值或最终 position RMSE 超过 `0.25 * rope length`；
- self-contact precision/recall/F1 和 topology accuracy（次指标；必须注明它们来自当前 simulator/proxy 标注）。

ID、每个 OOD tail 和 OOD macro average 分别报告。统计单位是 episode，不把 20 个时间步当作 20 个独立样本。

### 7.2 Counterfactual 主指标

- effect RMSE / rope length；
- relative effect error：`||effect_pred-effect_gt|| / (||effect_gt||+eps)`；
- effect cosine similarity；
- 每组五种材料最终形状的 pairwise-distance matrix error，避免只依赖一个 reference；
- 按 `K/E/rho/r` 分列，再做等权 macro average。

### 7.3 3 seeds 与不确定性

- 训练 seeds 固定为 `0, 1, 2`；三者使用完全相同的冻结数据 split。
- 主表给 3 个 seed 聚合值的 mean ± sample std，并在补充材料列出每个 seed。
- full 与 B1 的差异在相同 episode/group 上做 paired bootstrap（10000 次，episode 或 group 级重采样）并报告 95% CI。
- 不把 simulator/data seed 当作额外训练 seed，也不挑选“最好 seed”。

## 8. 训练和评估不变量

1. acceleration 是模型唯一运动学解码量，继续用 semi-implicit integration 得到 `vel_next/pos_next`；不另设独立 position-delta head。
2. 材料条件在 episode 内常量，但材料 encoder 输出要进入输入和每轮 processor。
3. rollout 第 `t+1` 步输入必须来自第 `t` 步预测；边由预测 `pos_t` 重建。
4. validation/test 禁止 teacher forcing，禁止使用 ground-truth contact pairs 构图。
5. failure label 继续由 tension/topology 派生，不变成独立真值。
6. 所有方法使用同一 action 表示、contact builder 和积分器；仅切换实验所声明的条件分支或 loss。
7. 检查 checkpoint 同时保存 config、normalizer、模型类型、seed 和数据 manifest hash。

## 9. 论文表格与图清单

### 主文必须完成

- Table 1：数据 split、ID/OOD 参数范围和 episode 数。
- Table 2：B0/B1/B2/B3 的 ID、OOD macro `position_NRMSE@1/@20`、divergence rate；完整 8 个 OOD tail 放附录。
- Table 3：A1/A2/A3 的材料 shuffle、注入位置和闭环训练消融。
- Table 4：四个材料轴的 counterfactual relative effect error 与 cosine。
- Figure 1：方法图。动态 state、episode material branch、每步 edge rebuild 分开画，并明确图中没有持久 SDF state。
- Figure 2：ID 与 OOD 的 horizon error curve，3 seeds mean ± std。
- Figure 3：K/E/rho/r 各 tail 的相对 B1 改善，不能只给合并柱状图。
- Figure 4：至少两个 paired group 的 simulator/full/B1 centerline overlay，并配五个 material scale 的 effect 曲线。

### 附录/补充材料

- 每 seed 完整数值、bootstrap CI、训练曲线、参数量与超参数；
- 数据 QA（初态/action 配对断言、参数直方图、NaN/爆炸率）；
- absolute-unit 指标和 normalized 指标；
- failure cases，尤其是接触重连、拓扑改变和高刚度发散。

## 10. 预先固定的 go / no-go 阈值

以下阈值在主实验前固定，防止看到结果后移动标准。

### GO：最小 claim 可进入投稿

所有 primary 条件都要满足：

1. **OOD 长时收益**：B3 相对 B1 在非 radius 的 6 个 `K/E/rho` OOD tails 上，`position_NRMSE@20` macro average 至少降低 10%，至少 5/6 slices 改善，且 3 个训练 seed 的 aggregate improvement 都为正。
2. **Counterfactual 收益**：在 `K/E/rho` 三轴 macro average 上，B3 相对 B1 的 relative effect error 至少降低 15%，effect cosine 至少提高 0.10；至少 2/3 参数轴分别改善。
3. **ID 非劣与稳定性**：B3 的 ID `position_NRMSE@20` 不得比 B1 高超过 5%；OOD divergence rate 不高于 B1 + 2 个百分点，且绝对值不超过 15%。
4. **条件确实被使用**：A1 shuffled-material 的 counterfactual relative effect error 相对正确材料输入至少恶化 5%。若只在 radius 上出现差异，不算通过，因为 radius 还参与构图。

同时满足以下 diagnostic 中至少一项：

- B3 相对 B2 的 OOD `position_NRMSE@20` 至少降低 5%，且 `@1` 不恶化超过 2%；
- B3 相对 B1 的 paired bootstrap 95% CI 在 OOD 主指标或 counterfactual 主指标上不跨 0。

### NO-GO / 必须收缩 claim

- 只有单步改善而 `@20` 无改善：不能写“long-horizon world model”；先修闭环训练或稳定性。
- OOD 改善但 material shuffle 不退化：模型可能没有使用显式条件，不能写材料条件 causal claim。
- 收益只来自 radius：先排除 contact builder 泄露/构图效应，最多写几何接触条件结果。
- paired-CF 初态或动作断言失败：整组数据作废并重生成，不能用统计修补。
- 三个 seed 中仅一个显著：报告不稳定性，不能只展示最好 seed。

若 primary 条件未全部满足，仍可投负结果/分析型 workshop 稿，但标题、摘要必须改成“诊断/基准”，不能沿用第 1.2 节的性能 claim。

## 11. 当前可执行的复现命令

生成器默认参数是小规模 MVP；论文协议必须显式传入本节数量。八个 OOD tail
和四个 counterfactual 参数分别放在独立目录，避免同名 split 被覆盖。生成器
默认拒绝覆盖，只有人工确认后才使用 `--force`。

```bash
# 1) ID 数据；三种 motion 按 episode 轮转
python scripts/gen_material_dataset.py \
  --out-dir runs/workshop_v1/id \
  --only train val test_id \
  --n-train 1200 --n-val 150 --n-test-id 300 \
  --motions random fold loop \
  --num-nodes 64 \
  --horizon 20 \
  --steps-interval 200 \
  --seed 0

# 2) 一个 OOD tail 示例；对 K/E/linear_density/radius × low/high 重复
python scripts/gen_material_dataset.py \
  --out-dir runs/workshop_v1/ood_K_low \
  --only test_ood_material --n-test-ood-material 75 \
  --ood-parameter K --ood-tail low \
  --motions random fold loop \
  --num-nodes 64 --horizon 20 --steps-interval 200 --seed 0

# 3) 一个 paired-CF 轴示例；对 K/E/linear_density/radius 重复
python scripts/gen_material_dataset.py \
  --out-dir runs/workshop_v1/cf_K \
  --only counterfactual --n-counterfactual 25 \
  --cf-parameter K --cf-scales 0.5 0.7 1.0 1.4 1.8 \
  --motions loop fold \
  --num-nodes 64 --horizon 20 --steps-interval 200 --seed 0

# 4) 同容量零材料 baseline；full model 改为 --model-type conditioned
python scripts/run_material.py \
  --data runs/workshop_v1/id/train.pt \
  --model-type conditioned_zero --seed 0 \
  --epochs 100 --val-horizon 20 --early-stop-patience 10 \
  --rollout-updates-per-epoch 4 --rollout-horizon 10 \
  --out runs/workshop_v1/conditioned_zero/seed_0.pt

# 纯单步 B2：--model-type conditioned --rollout-updates-per-epoch 0
# 原始小容量 B1：--model-type unconditioned

# 5) 对一个 checkpoint 和一个 OOD/CF 轴评估；输出含 correct/shuffled 条件
python scripts/eval_material.py \
  --checkpoint runs/workshop_v1/conditioned/seed_0.pt \
  --id-data runs/workshop_v1/id/test_id.pt \
  --ood-data runs/workshop_v1/ood_K_low/test_ood_material.pt \
  --counterfactual-data runs/workshop_v1/cf_K/counterfactual.pt \
  --out runs/workshop_v1/reports/conditioned_seed0_K_low.json
```

当前脚本已覆盖 B1、同容量 zero-condition、B2/B3、validation early stopping
和 ID/OOD/CF material shuffle。B0 constant-velocity、A2 input-only、八尾/三
seed 聚合和 paired bootstrap 仍是投稿前待实现项；在这些汇总入口完成前，不得把
单个 JSON 当成最终论文统计。

## 12. 投稿前剩余事项

### 代码与数据

- [x] 审查当前并行改动，保证单图/批图材料语义一致。
- [x] 在远端 DLO-Lab 环境跑 schema、serialization、provider、model backward 和 rollout 测试（当前 37 项通过）。
- [ ] 三个脚本的 `--help`、最小闭环和 manifest hash 已验证；仍需补长时生成的断点续跑/故障注入测试。
- [ ] 跑 24 条 smoke 数据并人工可视化 `random/fold/loop`；检查单位、张力尺度、接触率和拓扑标签分布。
- [ ] 生成冻结主数据；对 split seed、episode id、counterfactual group 做自动泄漏检查。
- [ ] 固定训练 config 后运行 B0/B1/B2/B3、A1/A2/A3 的 3 seeds。
- [x] 保存 raw per-episode metrics 与 shuffle assignment。
- [ ] 实现只从 raw JSON/CSV 生成表格、3-seed 汇总和 paired bootstrap 的入口。

### 科学叙事

- [ ] 在摘要和引言中把贡献限定为“已知材料条件 + simulated DLO rollout + paired intervention”。
- [ ] 补齐 GNS/DPI-Net、DLO dynamics、物理参数条件化、world models、counterfactual evaluation 相关工作。
- [ ] 解释为什么材料是 episode condition、接触边是逐步派生量，以及为什么 SDF 不属于动态 state。
- [ ] 报告 simulator/proxy tension、contact、topology 的局限，加入失败案例和负结果。
- [ ] 说明反事实中 `settle_steps=0` 是为控制初始条件，而普通 rollout 使用正常沉降；两者用途不同。

### 投稿工程

- [ ] 再次核对目标 workshop 的最新 CFP、页数、模板、双盲、non-archival 与截止时区。
- [ ] 完成 8 页内主文取舍；优先保留研究问题、协议、主结果、counterfactual 和限制。
- [ ] 清理作者名、仓库 URL、机器路径、checkpoint metadata 等双盲泄露。
- [ ] 固定图例、单位、有效数字和颜色；黑白打印与色弱模式检查。
- [ ] 至少一次由非作者按第 11 节从空目录复现 smoke run 和一条评估链路。

## 13. 执行顺序

1. 先通过 schema/provider/model 的远端测试和 24 条 smoke QA。
2. 冻结协议、数据 manifest 和 val-only 超参选择规则。
3. 生成全部数据并做泄漏/paired invariant 检查。
4. seed 0 跑 B1/B2/B3，确认数值稳定和指标导出正确；这一步只作工程检查，不据此改 go 阈值。
5. 配置冻结后运行 seeds 0/1/2 及所有必跑消融。
6. 自动生成表图与 bootstrap，按第 10 节做一次 go/no-go 判断。
7. 只有通过后才落笔性能 claim；否则收缩为诊断型论文并如实报告。
