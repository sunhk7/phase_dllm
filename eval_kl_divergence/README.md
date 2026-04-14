# 注意力与分布评估器工具箱 (Attention & Context Evaluation Toolbox)

本工具箱包含一系列精确的科学评估算法，旨在研究**局部信号充分性与全局注意力之间的权衡（Local Sufficiency vs Global Attention）**，专为深入验证扩散 / 掩码语言模型（例如 LLaDA）去噪过程的数学机理而打磨。通过底层逻辑魔改与多显卡并行脚本，极大化提高了模型机制验证的效率。

---

## 📂 底层模型基建映射

### `model/modeling_llada.py`
- **底层逻辑**：这是框架里底层的神经网络推断映射底座。我们在前向传播过程深处的 `_scaled_dot_product_attention` 内插入了自定针脚。通过无痛绕过 PyTorch 的原生 Block-Mask，我们强制组装了能够显式调控的任意形状“布尔张量路由网格（Boolean Routing Grids)”。
- **核心能力**：允许您随时发送诸如 `local_window_size=64` 来限制局部视野，更关键的是赋予了模型执行 `local_window_size={'w_size': 64, 'global_mask': is_easy}` 的绝活——这能强制把特定的空间位置打造为无论有多远都全局发散清晰波段的高维路标（Ground Truth Anchors）。

---

## 🛠️ 主力评估实验矩阵 (Evaluations & Bash Runners)

### 🔵 1. 标准散度衰减测算 —— “留一法”破坏容忍度
使用：`eval_kl_divergence.py` & 联合启动器与并行总线：`run_eval_kl_divergence.sh`
- **实验逻辑**：使用学术界最古典刻板的 Leave-One-Out（拔卜留一法）。在极具长度的历史长文中，程序每次抽走 **仅且绝对一个词** 变为未知（MASK）。它测算局部定长窗口面对仅这一个缺口时，由于缺少全景信息所引起的 Kullback-Leibler (KL) 解析发散程度。由于单句要重复抽上百次，验证开销极其昂贵扎实。
- **如何通过 sh 投递**：
  ```bash
  chmod +x run_eval_kl_divergence.sh
  ./run_eval_kl_divergence.sh
  ```
- **参数控制**：您可以直接在 Bash 脚本内部修改 `CONFIGS=("16" "32" "64" "16,64")`，运行时系统会自动把这四个繁重的遍历进程均分调度到不同的物理显卡。

---

### 🟢 2. 高噪状态下的动态拯救验证 —— SHiFT 效能定论验证
使用：`eval_mask_vs_clean.py` & 联合启动器与并行总线：`run_eval_mask_vs_clean.sh`
- **实验逻辑**：全图大面积致残法。直接施加巨量的随机掩码遮盖大半原句（生成 $z_t$ 去噪半途残状）。通过并行计算，脚本会在残损状态场上计算 **(Global 全局理想)**、**(Old Local 旧式盲猜)** 和 **(SHiFT 找路标的高级解法)**。脚本高度证明了：在长篇废墟下，如果有机制能动态甄别残境里的局部简单易存活词汇，其即可成为“强效路由节点”瞬间补回全境模型实力上限。
- **如何通过 sh 使用与传参**：
  ```bash
  chmod +x run_eval_mask_vs_clean.sh
  bash run_eval_mask_vs_clean.sh --mask_ratio 0.5 --use_shift --threshold 0.01
  ```
- **核心参数细节详解**：
  - `--mask_ratio [float]`：用来指定整块句子里有多少随机面积要被挖塌变成废墟（默认 `0.5` 即 $50\%$ 的 MASK 废墟掩码率）。
  - `--use_shift`：**核心特权 Flag**。启用后激活“第三路”前向计算。在废墟中动态用 Local 查出哪些是 $JSD < Threshold$ 的易算节点（天选骨干），再拿这些骨干充当 Landmark 全局内存重新喂养给 Local 重跑。最终渲染出双路悬殊反差的对比 CDF 折线。
  - `--threshold [float]`：控制上一步挑出易算骨干极严苛标准，只有被鉴定成 JSD 小于此阈值的才能被加冕成为全局特征锚点。

---

### 🟠 3. 单点骨干破局测验 —— The Origin Simulation
使用：`eval_shift_gt_simulation.py` & 联合启动器与并行总线：`run_eval_shift_gt_simulation.sh`
- **实验逻辑**：验证全局锚基点特性最初起家的祖师爷级探索。它通过极严标准的纯净扫描前置定位简单词（不依赖随机大面积造废墟），并使用全图只挖出一个空位的特化型精确测定，以此展现把收敛物当全局参考会带来的百分百数学等效修复力。
- **如何通过 sh 投递**：
  ```bash
  chmod +x run_eval_shift_gt_simulation.sh
  ./run_eval_shift_gt_simulation.sh
  ```
- **硬件分发**：天生包含了巨型的二维消融测试配置组网。代码利用 8 张显卡承载海量的细颗粒消融测点空间（阈值下限 vs 窗口滑差），跑完后自带生成非常华丽美观的聚合堆叠矩阵柱形图表组件。

---

### ⚪ 4. 无掩码无损对齐评估
使用：`eval_top1_agreement.py`
- **实验逻辑**：不施加甚至半分毫 MASK（在完好如初的 $z_0$ 无噪流形端），单独测定长文断距的影响。它探索如果在不挖掉信息的情况下生生切开对上文的绝对注意力连接，模型的最高词输出准心到底偏移几何（结论是毫不动摇）。
- **执行方式**（轻量秒跑工具，不设立 bash）：
  ```bash
  python eval_top1_agreement.py
  ```
