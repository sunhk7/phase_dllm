# 注意力与分布评估器工具箱

本工具箱包含几种精确的科学评估算法，旨在研究局部信号充分性与全局注意力之间的权衡（Local Sufficiency vs Global Attention），专为深入研究扩散 / 掩码语言模型（例如 LLaDA）而设计。它通过高吞吐量指标缓存，严格复制了分析条件（例如留一法去偏、Top-1 分布重叠）。

## 📁 `model/modeling_llada.py`
- **功能说明**：这是底层的神经网络映射。关键的注入发生在 `_scaled_dot_product_attention` 内部，我们绕过了原生的 PyTorch 块掩码（block-mask）处理，安装了显式的布尔张量路由网格（通过拦截 `local_window_size` 字典）。
- **核心验证**：实现了真实的 Shift-dLLM 的物理执行（即无论局部时间距离如何，作为 Ground Truth 的固定 token 都可以进行全局完整广播）。

---

## 📁 `eval_kl_divergence.py`
- **功能说明**：执行详尽的批量并行留一法（Token 掩码）提取模拟。
- **核心验证**：验证当用严格的滑动窗口注意力（$w$）统一替代全局自注意力时，对通用数据集 token 的 Kullback-Leibler (KL) 散度惩罚会发生什么变化。它量化了在纯粹的“盲文本上下文推断”期间产生的数学信号破坏。
- **如何独立执行**：
  ```bash
  python eval_kl_divergence.py --window 64
  ```

---

## 📁 `eval_shift_gt_simulation.py`  ⭐ SHiFT 皇冠上的明珠
- **功能说明**：动态执行精确的“Ground Truth 替换”模拟条件。
  1. 通过前置遍历识别出“简单 Token”（$KL < threshold$）。
  2. 将这些简单 Token 的空间边界无条件地以无掩码状态融入到注意力局部滑动掩码中（代表 `G_t`）。
  3. 严格针对剩余的“困难 Token”生成掩码（`masked`）排列，并追踪在通用滑动窗口与有 Shift-G_t 支持的边界条件下的预测崩溃行为。
- **核心验证**：从数学上验证了 SHiFT 背后的基础定理：暴露早期收敛特征可以完全消除随后的局部分布偏差，从而保障无限、安全的生成性能扩展。它绘制了由 Jensen-Shannon (JS) 散度构建的 CDF 映射，并汇总了专注于掩码失败案例的通用 Top-1 条形界限图。
- **如何独立执行**：
  ```bash
  python eval_shift_gt_simulation.py --window 32 --threshold 0.05
  ```

---

## 📁 `eval_top1_agreement.py`
- **功能说明**：评估在完全干净的标准上下文（$z_0$）下的纯类别对齐准确度。
- **核心验证**：作为一种理论上的界限验证，旨在探讨：如果所有的 token 身份都完全可见并且可读，丢弃历史 token 是否会显著影响 Top-1 Argmax 分类输出（提示：几乎没有任何影响）。
- **如何执行**：
  ```bash
  python eval_top1_agreement.py
  ```

---

## 📁 `run_eval_shift_gt_simulation.sh`
- **功能说明**：Bash 任务编排脚本。它将不同的消融实验超参数组合（Threshold 阈值 $\times$ 窗口区间）清晰地映射到 8 个独立 GPU 硬件的内存流中执行，最终通过执行一次合并聚合命令（`--plot_only`）来渲染跨接分布式任务的统一图表矩阵。
- **如何执行**：
  ```bash
  chmod +x run_eval_shift_gt_simulation.sh
  ./run_eval_shift_gt_simulation.sh
  ```
