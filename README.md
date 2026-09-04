# CALM 复现

论文: Contrastive learning for antibody-antigen sequence-to-specificity prediction (Lee et al., 2026)
DOI: 10.64898/2026.02.25.707916

本仓库复现论文的 Stage-1 对比学习部分（论文唯一训练并评测的部分），并实现 Stage-2 生成式 decoder 模块（论文只给架构、未训练，故本仓库同样只实现不训练）。

## 1. 复现范围

已实现并可运行:

- 从 SAbDab2 结构文件提取 CALM 所需的序列与界面掩码（BioPython 5 A 距离接触注释）
- 冻结式 embedding: 抗体 AntiBERTy (512 维, VH/VL 分别编码后拼接), 抗原 ESM-2 esm2_t33_650M_UR50D (1280 维)
- 双塔 FFN projection head + length-wise mean pooling + L2 归一化 + 余弦相似度 + 可学习温度
- 对称多正样本 InfoNCE 损失
- MMseqs2 同源性聚类切分 (40/60/80%) 与抗体聚类切分 (UC90/UC95), 随机错配基线 (RS)
- cluster-aware 嵌套 5-fold 交叉验证 (约 65:15:20)
- 双向 Recall@k (k=1,5,10) 评测, 候选池 256, 每折 512 个测试对
- 复现论文 Table 2/3/4 与 Figure 3/4
- 自回归 cross-attention decoder + task-prefix token (T5 风格), 含短过拟合自检

未实现（论文本身未做）:

- decoder 的正式训练与生成质量评测

## 2. 环境安装 (uv 一键)

服务器上只需要两条命令:

```bash
cd CALM-repro
bash scripts/setup_server.sh
```

脚本做的事情, 按顺序:

1. 没有 uv 就用官方安装脚本装到 `~/.local/bin`
2. `uv python install 3.11` + `uv sync --extra preprocess`, 按 `uv.lock` 在 `./.venv` 里装齐全部 Python 依赖（含 torch 2.5.1+cu124）
3. 尽力装 parasail, 装不上就退回 Biopython 比对后端并打印一行提示
4. 打印 `nvidia-smi` 里每张卡的型号 / 显存 / 驱动
5. 下载 mmseqs2 官方静态包到 `.tools/mmseqs`（按 `/proc/cpuinfo` 自动选 avx2 或 sse41）, 软链到 `.tools/bin`
6. 装 HMMER: 有 `sudo` 就走 `apt-get install hmmer`, 否则下载 hmmer 3.4 源码编译到 `.tools/hmmer`; 没有 C 工具链就跳过并告警
7. 跑 `scripts/check_env.py` 与 `calm info` 自检, 逐项打印 python / torch / cuda / 每张卡的 compute capability / mmseqs / hmmscan / 各依赖包

装完之后进入环境:

```bash
source scripts/env.sh
calm info
```

`scripts/env.sh` 把 `.venv/bin` 和 `.tools/bin` 加进 PATH, 并设置 `MPLCONFIGDIR`（否则 matplotlib 在无 HOME 写权限时会告警）。也可以不激活, 全程用 `uv run`:

```bash
uv run calm experiment --config configs/rtx4090/ag80.yaml
```

### 显卡适配

锁定的是 `torch==2.5.1+cu124`, 来自 `pyproject.toml` 里声明的 explicit 索引:

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu124", marker = "sys_platform == 'linux'" }]
```

cu124 轮子内置 sm_86 与 sm_89 两种 kernel, 分别对应 RTX 3060 与 RTX 4090, 两张卡都不需要额外编译。CUDA 运行时是向后兼容的, 目标机 `NVIDIA-SMI 595.84 / CUDA 13.2` 的驱动能直接跑 cu124 轮子。marker 限定 `sys_platform == 'linux'`, 所以在 macOS 上开发时会自动回落到 PyPI 的 CPU / MPS 版 torch, 同一个 `uv.lock` 两边都能用。

跑实验时用现成的两套配置选卡: `configs/rtx4090/`（`cuda:1`, batch 1024, bf16 autocast）与 `configs/rtx3060/`（`cuda:0`, batch 512）。

### 需要说明的三个依赖

- mmseqs2 必须安装。缺失时聚类会退化到精确序列匹配, 40/60/80 三档阈值会给出完全相同的结果, 论文的核心分层实验就失效了。程序会在 cluster_notes 里明确报警。
- ANARCI + AbNumber + HMMER 只在 `calm prepare` 阶段需要, 用于按 Chothia 编号从结构链里切出 VH/VL 可变区。如果直接用仓库里已生成的 `data/calm_dataset.csv`, 这三个可以不装。
- parasail 用于泄漏审计的 Needleman-Wunsch 比对。若 pip 装不上（例如 Python 3.13 无预编译轮子）, 本仓库会自动回退到 Biopython 的 PairwiseAligner, 使用同样的 BLOSUM62 / gap open 11 / gap extend 1 参数与同样的相似度定义（对齐区 percent identity 乘 alignment coverage）, 功能等价。

确认环境:

```bash
calm info
```

输出会报告 device、mmseqs2 是否可用、以及当前的 alignment backend。

## 3. 数据

论文的数据起点是 SAbDab 官方全量快照。请从 SAbDab 下载 "All structures" 归档并解压到 `data/sabdab_all_structures/`, 目录里需要包含:

- `sabdab_summary_all.csv` 官方元数据表 (22,149 行, 45 列), 每行是一个 Ab-Ag instance
- `pdb_<PDB_ID>/pdb_<PDB_ID>_sabdab.cif` 整条目结构文件 (11,543 个, 覆盖 summary 表里全部 PDB)

这份数据是结构文件 + 元数据表, 不是论文的 CALM schema 表, 所以本仓库写了适配层 `calm/sabdab.py`, 负责按论文口径筛选 instance、用 AbNumber 的 Chothia 编号从结构链里切出 VH/VL、按 5 A 距离标注 paratope/epitope、并输出论文所需的列。

```bash
calm prepare --summary-csv data/sabdab_all_structures/sabdab_summary_all.csv \
  --structure-dir data/sabdab_all_structures \
  --output data/calm_dataset_raw.csv
```

产出:

- `data/calm_dataset_raw.csv` 原始提取结果
- `data/calm_dataset.csv` 过滤后的训练用数据集
- `data/calm_dataset_raw.report.json` 逐级过滤统计

全量跑一次约 13 分钟 (按 PDB 分组, 每个 cif 只解析一次, 4,177 次结构解析)。本机实测的逐级统计:

| 阶段 | 行数 | 对应论文表述 |
|---|---|---|
| `sabdab_summary_all.csv` 输入 | 22,149 | |
| 有抗原链 | 19,802 | |
| 单一抗原链 | 9,164 | "only a single antigen chain", 排除多链界面 |
| 抗原类型为 protein | 7,312 | "antigen must be a protein" |
| 至少一条抗体链 | 7,312 | 至少有 H 或 L |
| 结构解析成功 | 7,216 | 96 行失败: 93 行无可变区, 3 行抗原链缺失 |
| 去重复 Ab-Ag 对后 | 5,299 | "duplicate pairs dropped" |
| 长度过滤后 (抗原 19-550 aa, 可变区 <=141 aa) | **4,326** | 论文 4,138 |

最终 **4,326 对 / 3,049 唯一抗体 / 3,450 唯一抗原**; 其中 VH+VL 双链 3,032, 仅 VH 1,275, 仅 VL 19; 抗原长度 50-550 aa, VH 最长 140 aa, VL 最长 117 aa; epitope 掩码 4,326/4,326 全部与抗原序列等长, 平均 20.91 个表位残基。条目类型分布 FAB 2,097 / SD-H 1,140 / FV 1,073 / SD-L 16。

有 13 行的 epitope 掩码全为 0 (结构里抗原与抗体在 5 A 内无接触)。这些行在 `mask_mode: interface` 的六个 regime 下由 `schema.validate(require_masks=True)` 自动剔除, 无需手工处理; 全长 regime 不受影响。

## 4. 与论文数据量的差异

论文报告 4,941 条原始记录过滤后得到 4,138 对 (3,332 唯一抗体 / 3,467 唯一抗原)。本复现得到 4,326 对 (3,049 / 3,450)。

| 指标 | 论文 | 本复现 | 偏差 |
|---|---|---|---|
| Ab-Ag 对 | 4,138 | 4,326 | +4.5% |
| 唯一抗体 | 3,332 | 3,049 | -8.5% |
| 唯一抗原 | 3,467 | 3,450 | -0.5% |
| VH+VL 双链 | 3,251 | 3,032 | -6.7% |
| 仅 VH | 874 | 1,275 | +45.9% |
| 仅 VL | 13 | 19 | +46.2% |

唯一抗原数几乎完全吻合 (-0.5%), 说明筛选口径与论文一致。剩余偏差来自数据快照版本:

- 本地 SAbDab 快照是 2026-08, 比论文使用的快照更新, 新增条目里纳米抗体 (single-domain, SD-H) 占比明显更高, 所以 "仅 VH" 一栏偏高 401 条, 这也是总对数反超论文的主要原因
- 唯一抗体数偏低而对数偏高, 说明这份快照里同一抗体对多个抗原的记录更多 (一对多), 与论文快照的冗余结构不同
- 论文未公布 4,941 条原始记录的具体来源列表, 无法逐条比对

结论: 管线口径与论文一致, 样本量差异来自 SAbDab 快照版本。绝对 Recall 数值因此不会与论文完全一致, 应重点比对趋势（40% < 60% < 80%, 掩码 > 全长, UC >> 同源性切分, RS 接近随机基线）。

旧版本的 SAbDab2 ML release 路径 (`--split-csv data/splits_final/abag_split.csv`) 仍然保留在 `calm/prepare.py` 里, 但那份切分表只覆盖 8,641 个 PDB, 最终只能得到 2,052 对, 不建议再用。

## 5. 复现命令

RTX 4090 (24 GB, device cuda:1, batch 1024):

```bash
for n in ag40 ag60 ag80 rs_full pa_ep40 pa_ep60 pa_ep80 rs_masked uc90 uc95; do
  calm experiment --config configs/rtx4090/\$n.yaml
done
calm figures --runs artifacts/runs --output artifacts/report
```

RTX 3060 (12 GB, device cuda:0, batch 512):

```bash
for n in ag40 ag60 ag80 rs_full pa_ep40 pa_ep60 pa_ep80 rs_masked uc90 uc95; do
  calm experiment --config configs/rtx3060/\$n.yaml
done
calm figures --runs artifacts/runs --output artifacts/report
```

两块卡都空闲时推荐把 10 个 regime 拆到两张卡上并行, 例如 4090 跑 pa_ep* 与 uc*, 3060 跑 ag* 与 rs_*。

embedding 缓存是 regime 无关的, 第一次运行会自动生成, 也可以先单独跑一次:

同一个 regime 可以安全重跑: mmseqs2 的临时目录每次会先清空再重建, results.json 直接覆盖, embedding 缓存则按 index.json 命中后复用（要强制重算用 `calm embed --config ... --overwrite`）。本机实测同一 regime 连续两次运行给出完全相同的 Recall 数值。

```bash
calm embed --config configs/rtx4090/ag40.yaml
```

只想看聚类和切分结果而不训练:

```bash
calm cluster --config configs/rtx4090/ag40.yaml
```

decoder 模块自检（含 300 步短过拟合, 证明模块可训练）:

```bash
calm decoder-check --steps 300
```

任意配置项都可以在命令行覆盖:

```bash
calm experiment --config configs/rtx4090/ag80.yaml --set train.epochs=50 --set evaluation.pool_size=128 --device cuda:0
```

## 6. 十个实验设置

| 配置 | 对应论文 | 掩码 | 聚类对象 | 聚类阈值 | 说明 |
|---|---|---|---|---|---|
| ag40 | Table 2 | 无 | 抗原 | 40% | Ab-Ag 全长 |
| ag60 | Table 2 | 无 | 抗原 | 60% | Ab-Ag 全长 |
| ag80 | Table 2 | 无 | 抗原 | 80% | Ab-Ag 全长 |
| rs_full | Table 2 | 无 | 抗原 | 80% | 随机错配基线 |
| pa_ep40 | Table 3 | 界面 | 抗原 | 40% | Pa-Ep 掩码 |
| pa_ep60 | Table 3 | 界面 | 抗原 | 60% | Pa-Ep 掩码 |
| pa_ep80 | Table 3 | 界面 | 抗原 | 80% | Pa-Ep 掩码 |
| rs_masked | Table 3 | 界面 | 抗原 | 80% | 随机错配基线 |
| uc90 | Table 4 | 界面 | 抗体 VH/VL | 90% | Unseen complex |
| uc95 | Table 4 | 界面 | 抗体 VH/VL | 95% | Unseen complex |

## 7. 输出产物

每个 regime 写入 `artifacts/runs/<name>/results.json`, 内含: 逐折 loss/lr/temperature/val 曲线、最优 epoch、双向与平均 test Recall@k、cluster backend 与 cluster_notes、数据过滤报告、随机基线。

`calm figures` 汇总到 `artifacts/report/`:

- `table2_full_length.csv` 论文 Table 2
- `table3_paratope_epitope.csv` 论文 Table 3
- `table4_unseen_complex.csv` 论文 Table 4
- `tables.md` 三张表的 markdown 版
- `figure3_recall.png` 论文 Figure 3 三面板 (R@1/R@5/R@10, 掩码橙色 / 全长蓝色, RS 斜线填充, 含 k/256 理论随机基线虚线, 误差棒为 95% CI)
- `figure4_unseen_complex.png` UC90/UC95 对比图
- `runs_index.json` 全部 run 的索引

## 8. 论文参考数值

以下是论文正文的点估计, 作为趋势比对基准（本复现数据量不同, 绝对值不会一致）。

| 设置 | 论文 R@1 | 论文 R@10 |
|---|---|---|
| Ab-Ag 全长 40% | 约 2% | 约 9% |
| Ab-Ag 全长 60% | 约 3% | 约 12% |
| Ab-Ag 全长 80% | 约 6% | 约 16% |
| Ab-Ag RS | 约 0.6% | 约 5.4% |
| Pa-Ep 掩码 40% | 约 2% | 约 11.8% |
| Pa-Ep 掩码 60% | 约 3% | 约 15% |
| Pa-Ep 掩码 80% | 约 7% | 约 19.4% |
| Pa-Ep RS | 约 0.8% | 约 6.3% |
| UC90 | 约 17.8% | 约 32.9% |
| UC95 | 约 19% | 约 35% |

候选池 256 时理论随机基线为 k/256: R@1 = 0.39%, R@5 = 1.95%, R@10 = 3.91%。

## 9. 关键实现决策

论文未写明或需要工程判断的地方, 这里逐条说明取舍。

掩码的实现方式。论文写的是 "二值掩码 + FFN projection + length-wise mean pooling"。本仓库采用 "先 gather 出被选中的 token, 再投影, 再对选中的 token 求平均", 而不是 "把未选中位置置零后对全长求平均"。原因: 池化后要做 L2 归一化, 所以除以选中数还是除以全长 L 对最终方向没有影响, 两者等价; 但如果先置零再送 FFN, 被掩掉的位置会产生 FFN(0) 这个非零常数并混入平均值, 反而污染表示。

投影与池化顺序。严格按论文: 先逐 token 投影, 再 length-wise mean pooling, 最后 L2 归一化。

Embedding 缓存格式。按唯一序列去重后存成 ragged fp16 memmap 加 index.json。抗原最长 550 aa、抗体 141 aa, 逐序列变长存储比 padding 成定长省一半以上空间, 且避免把 pad token 混进池化。

温度。可学习标量, 初始 0.2 (论文值), 以 log 形式参数化并 clamp 到 [0.01, 1.0] 防止训练早期塌缩。

checkpoint 选择。论文未指定。本仓库用验证集 Recall@10 选最优 epoch。

聚类回退策略。mmseqs2 缺失时回退为 `Cluster_id` 列（来自数据集 agclusters）, 抗体侧回退为 VH/VL 精确序列匹配, 并在 cluster_notes 中写明 "identity backend cannot honour distinct split and subsample identity thresholds"。请务必安装 mmseqs2。

泄漏审计。mmseqs2 是启发式聚类, 可能把高度同源的序列分到不同簇。所以在 mmseqs2 之后再补一道兜底: 对每个聚类取最长序列作代表, 两两比对, 相似度 = 对齐区 percent identity 乘 alignment coverage; 超过阈值的簇对用并查集合并, 从而保证切分后跨折不存在高相似样本。代表数上限 3000, 超限则跳过并在 notes 中告警。

这一步是 O(n^2), 在 4,326 对数据集上有 952 个抗原簇代表, 即 452,676 次比对, Biopython 后端下要跑约 40 分钟。因此加了一道精确剪枝: 相似度按定义等于 `matches / max(len_a, len_b)`, 而全局比对的 `matches <= min(len_a, len_b)`, 所以相似度上界就是 `min(len) / max(len)`。长度比例本身低于阈值的簇对不可能达标, 直接跳过。这个界是紧的, 结果与不剪枝完全一致。实测在 80/60/40 三档分别把比对数压到 97,316 / 142,533 / 158,811 次 (剩余 21.5% / 44.7% / 70.0%)。

审计结果本身很干净: ag80 在 952 个代表里只额外合并 1 对, ag40 在 674 个代表里合并 1 对, uc90 在 1,049 个代表里合并 0 对。说明 mmseqs2 的聚类没有漏掉明显同源对, 也没有出现单链接链式坍缩。

显存策略。encoder 全程冻结, 只训练两个 projection head。4090 用 batch 1024 + bf16 autocast + token_chunk 262144; 3060 用 batch 512 + token_chunk 65536。训练时对分块投影使用 `torch.utils.checkpoint(use_reentrant=False)` 压峰值显存, 仅在 CUDA 上启用。

CDR 定义。数据集提供的是 IMGT 编号, 而论文用 Chothia。列名沿用 `*_Chothia_Mask` 以保持 schema 兼容, 但这些列不参与训练与评测, 只作为附带注释。

decoder 初始化。`nn.Embedding` 默认 N(0,1) 加上输出层权重绑定会让初始 loss 高达 40, 远离 ln(26)=3.26 的理论值。已改为 std=0.02 初始化, 实测初始 loss 3.25, 300 步过拟合到 0.003 并能完整重建输入序列。

## 10. 已知偏差

- 样本量 4,326 对 vs 论文 4,138 对, 原因见第 4 节
- 数据源是 2026-08 的 SAbDab 官方快照, 不是论文使用的快照; 纳米抗体条目更多, 导致 "仅 VH" 样本偏高
- UC 设置 (`cluster_on: antibody`) 下最大簇有 1,041 对 (占 31%)。原因是同一样本的 VH 与 VL 分别落在不同 mmseqs2 簇时会被并查集合并, 产生链式聚合。这是 cluster-aware 切分的固有性质, 论文的 UC 设置同样存在, 但会让 5 折的折间样本量不均衡
- CDR 注释用 IMGT 而非 Chothia（不影响训练）
- 泄漏审计默认走 Biopython 而非 parasail（相似度定义与比对参数一致）
- decoder 只实现与自检, 不训练, 与论文一致

## 11. 目录结构

```
calm/
  cli.py      命令行入口 (prepare/embed/cluster/train/experiment/figures/decoder-check/info)
  sabdab.py   SAbDab 官方全量快照到 CALM schema 的适配层 (主路径), AbNumber Chothia 编号, 5 A 界面标注
  prepare.py      SAbDab2 ML release 切分表的旧适配层 (保留, 已不推荐)
  schema.py       列定义、过滤规则、数据校验
  embeddings.py   AntiBERTy / ESM-2 冻结 embedding 与 ragged memmap 缓存
  dataset.py      掩码 token 选取与 batch 组装
  clustering.py   MMseqs2 聚类、比对泄漏审计、每簇下采样、随机错配
  splits.py       cluster-aware 嵌套 5-fold 切分与不相交断言
  model.py        双塔 projection head、池化、可学习温度
  losses.py     对称多正样本 InfoNCE
  metrics.py      双向 Recall@k、随机基线、95% CI
  train.py        AdamW + CosineAnnealingWarmRestarts 训练循环
  evaluate.py     候选池采样与评测
  experiment.py   单 regime 的完整嵌套交叉验证流程
  figures.py    Table 2-4 与 Figure 3/4 渲染
  decoder.py      自回归 cross-attention decoder 与 task-prefix tokenizer
configs/          10 个 regime 的基线配置
configs/rtx3060/  RTX 3060 12 GB 适配 (cuda:0, batch 512)
configs/rtx4090/  RTX 4090 24 GB 适配 (cuda:1, batch 1024)
configs/smoke/    CPU 冒烟配置 (hashed encoder, 缩减超参, 用于快速验证管线)
scripts/
  setup_server.sh 服务器一键环境安装 (uv + mmseqs2 + HMMER + 自检)
  env.sh          激活脚本 (PATH / VIRTUAL_ENV / MPLCONFIGDIR)
  check_env.py    环境自检 (python / torch / cuda / 每张卡 compute capability / 外部二进制 / 依赖包)
tools/            配置生成器与 PDF 素材提取脚本
pyproject.toml    依赖声明与 torch cu124 explicit 索引
uv.lock           锁定的依赖版本 (linux x86_64 与 darwin arm64 两套 marker)
```

## 12. 快速验证

没有 GPU 或不想下载 ESM-2 权重时, 可以用 `configs/smoke/` 跑完整管线。这套配置把 encoder 换成确定性的 hashed featurizer, 候选池缩到 32, epoch 缩到 30, 单个 regime 在 CPU 上约两分钟。

```bash
for n in ag40 ag60 ag80 rs_full pa_ep40 pa_ep60 pa_ep80 rs_masked uc90 uc95; do
  calm experiment --config configs/smoke/\$n.yaml
done
calm figures --runs artifacts/smoke/runs --output artifacts/smoke/report
```

冒烟配置的绝对数值没有生物学意义（encoder 是随机投影特征）, 它验证的是: 聚类分层生效、切分无泄漏、loss 下降、recall 高于随机基线、出图出表正常。
