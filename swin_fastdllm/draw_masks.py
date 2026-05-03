import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
import numpy as np


plt.rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
})

def draw_attention_mask(ax, core_size=32, window_size=8, pad_size=0, title="", window_colors=None, show_y_axis=False, core_color_layout=False):
    """
    绘制单层的 Attention Mask 矩阵
    :param core_size: 核心 Token 数量 (D=32)
    :param window_size: 局部窗口大小 (W=8)
    :param pad_size: 边缘 Padding 大小 (Shift=4 时，Pad=4)
    """
    # 科研绘图常用配色 (柔和且对色弱友好)
    c_white = "#FFFFFF"
    c_green = "#A8D08D"  # 局部可见窗口 (绿色)
    c_blue  = "#4A86E8"  # 对角线/自注意力 (蓝色)
    c_gray  = "#B5B5B5"  # Padding 屏蔽区 (灰色)
    c_line  = "#888888"  # 网格线颜色
    
    grid_size = core_size + pad_size * 2
    window_starts = list(range(0, grid_size, window_size))
    
    # 设置坐标轴
    ax.set_xlim(0, grid_size)
    ax.set_ylim(grid_size, 0) # 反转 Y 轴，使 (0,0) 在左上角，符合矩阵习惯
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(show_y_axis)
    ax.spines['left'].set_visible(show_y_axis)
    ax.tick_params(axis='both', which='both', length=0, labelsize=10)
    if show_y_axis:
        tick_positions = np.arange(0, grid_size + 1, window_size)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(int(t)) for t in tick_positions])
        ax.set_ylabel('Token ID', fontsize=13, fontweight='bold')
        ax.set_xticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # 1. 铺设白色底板
    ax.add_patch(patches.Rectangle((0, 0), grid_size, grid_size, facecolor=c_white))

    # 2. 绘制绿色可见窗口
    # 对于带有 padding 的情况，窗口起始点从 0 开始，按 window_size 步进
    if window_colors is None:
        window_colors = [c_green] * len(window_starts)

    # 在 shifted 层中可保持与 Layer L 相同的核心配色结构，让一个 shifted window 横跨两个颜色块
    color_starts = window_starts
    if core_color_layout and pad_size > 0:
        color_starts = list(range(pad_size, pad_size + core_size, window_size))

    for idx, start in enumerate(color_starts):
        color = window_colors[idx % len(window_colors)]
        ax.add_patch(patches.Rectangle(
            (start, start), window_size, window_size,
            facecolor=color, edgecolor='none'
        ))

    # 3. 绘制灰色的 Padding 遮罩层 (覆盖在绿色窗口上)
    if pad_size > 0:
        # 顶部和底部 Padding (横向宽条)
        ax.add_patch(patches.Rectangle((0, 0), grid_size, pad_size, facecolor=c_gray, edgecolor='none'))
        ax.add_patch(patches.Rectangle((0, grid_size - pad_size), grid_size, pad_size, facecolor=c_gray, edgecolor='none'))
        # 左侧和右侧 Padding (纵向窄条)
        ax.add_patch(patches.Rectangle((0, 0), pad_size, grid_size, facecolor=c_gray, edgecolor='none'))
        ax.add_patch(patches.Rectangle((grid_size - pad_size, 0), pad_size, grid_size, facecolor=c_gray, edgecolor='none'))

    # 4. 绘制蓝色主对角线 (只在非 Padding 区域绘制)
    for i in range(grid_size):
        if pad_size <= i < grid_size - pad_size:
            ax.add_patch(patches.Rectangle((i, i), 1, 1, facecolor=c_blue, edgecolor='none'))

    # 5. 绘制精细网格线 (1x1 Token 级别)
    for i in range(grid_size + 1):
        ax.plot([0, grid_size], [i, i], color=c_line, lw=0.3, zorder=3) # 横线
        ax.plot([i, i], [0, grid_size], color=c_line, lw=0.3, zorder=3) # 竖线

    # 6. 绘制粗的黑色窗口边界线
    for start in window_starts:
        ax.add_patch(patches.Rectangle(
            (start, start), window_size, window_size, 
            fill=False, edgecolor='black', lw=1.5, zorder=4
        ))
        
    # 7. 绘制整个大矩阵的外边框
    ax.add_patch(patches.Rectangle(
        (0, 0), grid_size, grid_size, 
        fill=False, edgecolor='black', lw=2.0, zorder=5
    ))

    # 设置标题
    ax.set_title(title, fontsize=16, pad=15, fontweight='bold', fontfamily='Times New Roman')


# ================= 配置与生成画布 =================
gradient_colors = plt.cm.GnBu(np.linspace(0.35, 0.85, 4))

# 创建画布，2 行 5 列，第一行放 Sequence Block，第二行放 Attention Masks
# 按真实网格尺寸分配宽度，让 40x40 的 Layer L+1 明显大于 32x32 的两侧图
fig = plt.figure(figsize=(24, 11))
gs = fig.add_gridspec(2, 5, width_ratios=[32, 2, 40, 2, 32], height_ratios=[1, 3.5], wspace=0.05, hspace=0.3)

# --- 绘制顶层的 Sequence Blocks ---
ax_top = fig.add_subplot(gs[0, :])
ax_top.set_xlim(0, 10)
ax_top.set_ylim(0, 2)
ax_top.axis('off')

n_blocks = 5
block_w = 1.2
block_h = 0.8
start_x = (10 - n_blocks * block_w) / 2
current_idx = 2  # 中间这个为 Current Block

for i in range(n_blocks):
    x = start_x + i * block_w
    y = 0.6
    if i in [0, 1]:  # Prefix (块0, 块1)
        facecolor = '#DDEBF7'
        edgecolor = '#5C8ECA'
        lw = 1.5
        zorder = 2
    elif i == current_idx:  # Current
        facecolor = '#FFE699'
        edgecolor = '#D6B656'
        lw = 2.5
        zorder = 5
    else:  # Future
        facecolor = '#F5F5F5'  
        edgecolor = '#CCCCCC'
        lw = 1.5
        zorder = 2
    
    ax_top.add_patch(patches.Rectangle((x, y), block_w, block_h, facecolor=facecolor, edgecolor=edgecolor, lw=lw, zorder=zorder))

# 添加图外标注 (放于块上方)
# Prefix 标注
prefix_center_x = start_x + block_w  # 跨过块0和块1 的中间
ax_top.text(prefix_center_x, 1.65, 'Prefix', ha='center', va='center', 
            fontsize=16, fontfamily='Times New Roman', color='#3C78B9', fontweight='bold')
ax_top.plot([start_x, start_x + 2 * block_w], [1.48, 1.48], color='#5C8ECA', lw=3.0)

# Current Block 标注
current_center_x = start_x + current_idx * block_w + block_w / 2
ax_top.text(current_center_x, 1.65, 'Current Block', ha='center', va='center', 
            fontsize=16, fontfamily='Times New Roman', color='#CCA026', fontweight='bold')
ax_top.plot([start_x + current_idx * block_w, start_x + (current_idx + 1) * block_w], [1.48, 1.48], color='#D6B656', lw=3.0)

cb_x_left = start_x + current_idx * block_w
cb_x_right = cb_x_left + block_w
cb_y_bottom = 0.6

# --- 绘制下方的 Attention Masks ---
axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 4])]
arrow_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 3])]

for arrow_ax in arrow_axes:
    arrow_ax.set_axis_off()
    arrow_ax.annotate(
        '',
        xy=(0.95, 0.5), xytext=(0.05, 0.5),
        arrowprops=dict(arrowstyle='->', lw=2.5, color='#555555')
    )

# Layer L: 无 Padding，完全对齐的 32x32
draw_attention_mask(
    axes[0], core_size=32, window_size=8, pad_size=0, 
    title="Layer L",
    window_colors=gradient_colors,
    show_y_axis=True,
)

# Layer L+1: Shift = 4，所以两侧各加 4 的 Padding，总图幅 40x40
draw_attention_mask(
    axes[1], core_size=32, window_size=8, pad_size=4, 
    title="Layer L+1",
    window_colors=gradient_colors,
    show_y_axis=False,
    core_color_layout=True,
)

# Layer L+2: 回归到 Layer L 的状态
draw_attention_mask(
    axes[2], core_size=32, window_size=8, pad_size=0, 
    title="Layer L+2",
    window_colors=gradient_colors,
    show_y_axis=False,
)

# --- 增加放大连线 (Zoom in 效果) ---
con1 = ConnectionPatch(xyA=(cb_x_left, cb_y_bottom), xyB=(0, 0), coordsA="data", coordsB="data",
                       axesA=ax_top, axesB=axes[0], color="#D6B656", ls="--", lw=2.5, alpha=0.8)
# 因为图的Y轴是反向的(0在上)，所以B的坐标是 (32, 0)
con2 = ConnectionPatch(xyA=(cb_x_right, cb_y_bottom), xyB=(32, 0), coordsA="data", coordsB="data",
                       axesA=ax_top, axesB=axes[2], color="#D6B656", ls="--", lw=2.5, alpha=0.8)
fig.add_artist(con1)
fig.add_artist(con2)

plt.tight_layout()

# 导出为高清晰度图片 (论文推荐存为 .pdf 矢量图)
# plt.savefig("attention_mask_evolution.pdf", format='pdf', bbox_inches='tight')
# plt.savefig("attention_mask_evolution.png", dpi=300, bbox_inches='tight')

plt.show()