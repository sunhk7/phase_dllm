【Context】
我正在研究 Diffusion LLM (LLaDA-8B) 的全并行解码 (Jacobi-style decoding) 推理加速。我需要绘制一张“时空动力学演化热力图” (Spatio-Temporal Dynamics Heatmap)，以证明大多数 Token 只需要局部搜索，不需要全局 Attention。

我已经通过本地导入成功接管了模型：
`from model.modeling_llada import LLaDAModelLM`

现在，请你帮我完成代码的无缝注入与数据收集可视化。为了不破坏 Hugging Face 复杂的返回值结构，我们将使用给 `model` 实例动态绑定属性的方式来收集数据。

【Task 0: 编写独立的极简测试脚本 `test_dummy_model.py`】
在修改真实的 `generate.py` 之前，先写一个玩具脚本跑通数据闭环：
1. 导入 `model.configuration_llada` 的 Config 类，实例化一个极小的随机权重模型 (num_hidden_layers=4, hidden_size=256, num_attention_heads=8)，转到 cuda 和 bfloat16。不要调用 from_pretrained。
2. 为这 4 层 Attention 挂载 `global_ratio_tracker = []`。
3. 伪造 shape 为 (1, 128) 的随机 input_ids，放入 for 循环模拟 10 步生成，收集 tracker 数据，存为 `dummy_dynamics.npy`，并调用画图代码。

【Task 1: 在 `model/modeling_llada.py` 中注入局部性测算】
在 `Attention` 类的 `forward` 函数中，计算完 softmax 得到 `attn_weights` 后：
1. 定义局部窗口大小 W=64（对角线前后各 32）。
2. 使用纯 PyTorch 张量操作 (`torch.triu`, `torch.tril`) 将对角线 ±32 范围内的权重 mask 为 0。绝对不能用 for 循环。
3. 对剩余权重在最后维度求和，并求全 Batch、Heads、Seq_Len 的平均值，得到标量 `Global_Weight_Ratio`, 即分配在局部窗口之外的注意力权重只和
4. 核心注入：`if hasattr(self, 'global_ratio_tracker'): self.global_ratio_tracker.append(Global_Weight_Ratio)`。

【Task 2: 在现有的 `generate.py` 中收集真实数据】
在我的 LLaDA-8B 全并行解码步骤 (steps=64) 中：
1. 循环开始前，遍历 `model.model.layers`，给每个 attention 模块挂载 `layer.self_attn.global_ratio_tracker = []`。
2. 每步迭代结束时，提取所有层 tracker 的最后一个值，存入矩阵。
3. 循环结束后，保存为 `llada_8b_attention_dynamics.npy`。

【Task 3: 编写独立的极简测试脚本 `test_dummy_model.py`】
为了极速验证上述修改是否正确，请写一个独立的极简测试脚本：

1. 从 model.configuration_llada 实例化一个极小的随机权重配置 (num_hidden_layers=4, hidden_size=256, num_attention_heads=8)，构建 dummy_model，转入 cuda 和 bfloat16。不要调用 from_pretrained。
2. 导入加载真实的 Tokenizer。
3. 导入我修改好的生成函数：from generate import generate。
4. 编码一段极简 Prompt，调用 generate(dummy_model, input_ids, tokenizer, steps=10, gen_length=32, block_length=16, ...)。
5. 这个脚本的目的是确保 modeling_llada 的张量计算和 generate 的数据收集不报错，并成功输出一个 10x4 的 .npy 文件。

【Task 4: 编写独立的可视化脚本 `plot_dynamics.py`】
1. 读取传入的 `llada_8b_attention_dynamics.npy` 矩阵。
2. 用 `seaborn` 绘制热力图。X轴为 Diffusion Timestep，Y轴为 Transformer Layers。
3. 颜色映射必须使用 `cmap='coolwarm'`，低比例显示为冷色调（局部搜索），高比例显示为暖色调（全局搜索）。添加 Colorbar 和学术标题，保存为高清 PNG。

【输出要求】
请分别给出 `test_dummy_model.py` 的完整代码、`modeling_llada.py` 中需要插入的核心逻辑片段、`generate.py` 的数据收集片段，以及 `plot_dynamics.py` 的完整代码。