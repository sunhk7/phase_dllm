【背景与目标】
我正在研究 Diffusion LLM (dLLM) 的全并行解码推理加速，核心思想是基于“相变 (Phase Transition)”概念：在推理过程中，信息熵低的 Token 已经处于稳定的“固态”，只需要局部 Attention；而信息熵高的 Token 仍处于“液态”，需要全局 Attention。我们计划结合 Q-sparsity、KV-sparsity 和 KV cache 管理来实现这一顶会级别的加速方法。

【目录结构与隔离要求】
目前我有一个名为 `experiments` 的目录，里面包含：
- `__init__.py `
- `experiments/generate.py`
- `experiments/model/configuration_llada.py`
- `experiments/model/modeling_llada.py`

**核心操作要求**：将上述三个基础文件完整复制到独立的新目录 `experiments_entropy` 中（保持相同的子目录结构，如 `experiments_entropy/model/`）。后续所有的代码修改和数据收集，都必须在这个新目录下的副本文件中进行，绝对**不可以**修改原 `experiments` 目录下的任何文件。

【硬件约束】
运行环境为单卡 RTX 3090 (24GB VRAM)。全并行解码极其消耗显存，代码必须严格遵守显存管理规范（使用 torch.inference_mode()、及时使用 del 释放大张量、必须将需要保存的统计数据 `.cpu().numpy()` 化）。

【任务要求】
请帮我自动完成文件的创建与复制后，执行以下代码编写与修改任务：

### Task 1: 修改副本文件并编写 `collect_entropy_data.py`
在你刚刚创建并复制好的 `experiments_entropy` 目录中：

1. **修改 `experiments_entropy/model/modeling_llada.py`**：
   在 Attention 计算完成得到 `attn_weights` 后：
   - 设定局部窗口 W=64（对角线 ±32）。
   - 将 Attention 矩阵的 Q 维度在 `context_length` 处切片，只提取生成的 Token。
   - Mask 掉局部窗口后，在 Key 维度 (dim=-1) 求和，在 Heads 维度 (dim=1) 求平均，得到 shape 为 `(batch_size, generated_seq_len)` 的 1D 张量。
   - 转为 numpy 并存入 `layer.self_attn.global_ratio_tracker`。使用 `del` 及时清理张量。

2. **修改 `experiments_entropy/generate.py`**：
   在 64 步的生成循环中：
   - 每次模型 forward 输出 `logits` 时，切取出生成部分的 `gen_logits`。
   - 计算这些 Token 的预测信息熵（公式：$H = -\sum P \log(P + 1e-9)$）。
   - 提取模型第 24 层最新的 `global_ratio_tracker` 数据。
   - 将每一步的 Entropy (1D array) 和 Global Ratio (1D array) 结对配对。
   - 循环结束后，将所有配对数据保存为 `entropy_ratio_pairs.npy`。

3. **编写并生成 `experiments_entropy/collect_entropy_data.py`**：
   - 导入修改好的本地副本：`from model.modeling_llada import LLaDAModelLM` 和 `from generate import generate`。
   - 使用 `datasets` 库加载 `wikitext-103-v1`，截取一段长文本。
   - Tokenize 后切分为 `prompt_length=256`。
   - 初始化一个极小的 Dummy Model（4层，hidden=256），调用 `generate` 函数，设置 `gen_length=768`，`steps=64`，跑通整个数据闭环。

### Task 2: 编写并生成 `experiments_entropy/plot_entropy_kde.py` (高清核密度可视化)
读取上一步生成的 `.npy` 数据（约 64 * 768 = 49152 个数据点），绘制一张高质量的二维学术图表。
1. **使用 Seaborn**：必须使用 `sns.kdeplot` (二维核密度图，推荐 `fill=True, cmap="mako"`) 或 `sns.jointplot(kind="hex")`，切勿使用普通散点图。
2. **坐标轴设置**：
   - X 轴：Token Prediction Entropy (预测信息熵)
   - Y 轴：Global Attention Weight Ratio (全局注意力比例)
3. **学术美化**：
   - 添加清晰的 Title、X/Y Label 和 Colorbar。
   - 标注出“Solid Phase (Low Entropy, Local)”和“Liquid Phase (High Entropy, Global)”的物理预期区域。
   - 保存为 `phase_transition_kde.png`，dpi=300。

### Task 3: 模仿 experiments/ 创建 commands.sh 和 config.yaml 进行管理

【输出要求】
完成后台的文件操作后，请向我展示：
1. `modeling_llada.py` 中注入的核心代码片段（请标明精确的上下文插入位置）。
2. `generate.py` 中注入的数据收集片段（请标明精确的上下文插入位置）。
3. `collect_entropy_data.py` 和 `plot_entropy_kde.py` 的完整代码，包含详尽的中文注释。