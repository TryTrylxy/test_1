import matplotlib.pyplot as plt
import numpy as np
import ast

# 初始化存储所有dict的列表
result_list = []

# 读取txt文件并解析
with open("/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_07/output/vitb32/vg/cusa_restate_sl_seed23/log_179/tripletlossrate_ab.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # 去除换行/空格
        if line.startswith("result:"):
            # 截取result:后的字典字符串并解析
            dict_str = line.split("result:", 1)[1].strip()
            try:
                # 安全解析字符串为字典（比eval更安全）
                data_dict = ast.literal_eval(dict_str)
                result_list.append(data_dict)
            except (SyntaxError, ValueError):
                # 跳过格式错误的行，避免程序崩溃
                print(f"跳过格式错误的行：{line}")
result_list = result_list
# 提取数据
a = [x['r_sum'] for x in result_list]
b = [x['text_neg'] for x in result_list]
c = [x['image_neg'] for x in result_list]
x = np.arange(0.1, 1.1, 0.1)  # x轴

# 定义论文常用色值
color_a = '#1f77b4'  # 深蓝（Rsum）
color_b = '#d62728'  # 猩红（ACC_T）
color_c = '#2ca02c'  # 墨绿（ACC_I）

# 创建画布和主坐标轴（左纵轴）
fig, ax1 = plt.subplots(figsize=(8, 5))

# 绘制a折线（左纵轴）
ax1.plot(x, a, color=color_a, marker='o', label='Rsum', linewidth=1.5)
ax1.set_xlabel(r'$\alpha$', fontsize=12) 
ax1.set_ylabel('Rsum', color=color_a, fontsize=10)
ax1.tick_params(axis='y', labelcolor=color_a)

# 创建右侧纵轴，共享x轴
ax2 = ax1.twinx()
# 绘制b、c折线（右纵轴）
ax2.plot(x, b, color=color_b, marker='s', label=r'$ACC_T$', linewidth=1.5)
ax2.plot(x, c, color=color_c, marker='^', label=r'$ACC_I$', linewidth=1.5)

# 核心修改：右侧纵坐标标签用b+c颜色融合（渐变/拼接），刻度分色
# 方法1：标签文字用双色拼接（论文常用，简洁易实现）
ax2.set_ylabel(r'$\mathrm{ACC_T}$ / $\mathrm{ACC_I}$', fontsize=10, 
               color='#800080')  # 紫色作为标签底色（中性色，兼容红/绿）
# 方法2：刻度标签分色（b对应红，c对应绿，更精准）
ax2.tick_params(axis='y', labelcolor='black')  # 刻度值设为黑色（清晰）
# 若需更精细：分别设置刻度颜色（可选）
# ax2.spines['right'].set_color((color_b[:3] + color_c[3:]))  # 轴边框渐变（进阶）

# 合并图例并放在中间上方
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

# 调整布局（避免图例被截断）
plt.tight_layout()
# 保存高清图片（论文级300dpi）
plt.savefig('three_lines.png', bbox_inches='tight', dpi=300)