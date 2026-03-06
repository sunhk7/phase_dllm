# 实验思路（LLaDA 全并行解码局部性）

## 目标
验证在 Jacobi-style 全并行解码过程中，多数 token 的注意力主要集中在局部窗口内（对角线附近），并通过时空热力图展示该动态。

## 核心假设
- 在 diffusion timestep 的大部分阶段，跨远距离 token 的全局注意力占比更低。
- 不同层的全局注意力占比存在结构性差异（浅层/深层行为不同）。

## 指标定义
- `Global_Weight_Ratio`：每层每步中，attention 在局部窗口外的权重和（对 batch/head/seq 求平均后的标量）。
- 局部窗口：`W=64`（对角线前后各 32）。

## 实验设计
- Dummy 验证：4 层随机小模型，确保张量计算、数据收集、保存与画图链路正确。
- Real 运行：LLaDA-8B-Instruct 真实推理，收集全步骤全层比率矩阵。

## 结果产物
- `results/dummy_dynamics.npy`
- `results/dummy_dynamics.png`
- `results/llada_8b_attention_dynamics.npy`
- `results/llada_8b_attention_dynamics.png`

## 关注点
- 观察热力图中冷色（低全局占比）是否占主导。
- 对比不同层在中后期 timestep 是否出现局部->全局的迁移。
- 若出现大面积暖色，检查 prompt 长度、mask 调度、remasking 策略。
