import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable


# with open("fakeedit_real_logits.txt", 'r', encoding='utf-8') as file:
#     real_logits = file.read()
#     # file.write(str(real_logits))
# with open("fakeedit_fake_logits.txt", 'r', encoding='utf-8') as file:
#     fake_logits = file.read()
#     # file.write(str(fake_logits))
# with open("fakeedit_labels.txt", 'r', encoding='utf-8') as file:
#     labels = file.read()
#     # file.write(str(labels))
# best_f1 = 0
# best_acc = 0
# best_pre = 0
# best_rec = 0
# # print(real_logits)
# # print(type(real_logits))
# # print(len(real_logits))
# real_logits = ast.literal_eval(real_logits)
# fake_logits = ast.literal_eval(fake_logits)
# labels = ast.literal_eval(labels)
# print(len(real_logits))
# print(len(fake_logits))
# print(len(labels))
# X = []
# Y = []
# Z = []
# for a in range(0, 11):
#
#     best_f12 = 0
#     best_acc2 = 0
#     best_pre2 = 0
#     best_rec2 = 0
#     best_parameters2 = {}
#     for b in range(0, 11 - a):
#         X.append(a / 10.0)
#         Y.append(b / 10.0)
#         c = 10 - a - b
#         tp = 0
#         tn = 0
#         fp = 0
#         fn = 0
#         for x in range(len(real_logits)):
#             # aerfa = beita = sigema = 0.33
#             aerfa = a / 10.0
#             beita = b / 10.0
#             sigema = c / 10.0
#             final_real_logits = aerfa * real_logits[x][0] + beita * real_logits[x][1] + sigema * real_logits[x][2]
#             final_fake_logits = aerfa * fake_logits[x][0] + beita * fake_logits[x][1] + sigema * fake_logits[x][2]
#             label = labels[x]
#             final_label = "real" if final_real_logits > final_fake_logits else "fake"
#             if final_label == label:
#                 if label == "real":
#                     tp = tp + 1
#                 else:
#                     tn = tn + 1
#             else:
#                 if label == "fake":
#                     fp = fp + 1
#                 else:
#                     fn = fn + 1
#         acc = (tp + tn) / (tp + tn + fp + fn)
#         pre = tp / (tp + fp)
#         rec = tp / (tp + fn)
#         f1 = 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / (tp / (tp + fp) + tp / (tp + fn))
#         Z.append(f1)
#         # print(f"text:{aerfa},image:{beita},mm:{sigema}acc:{acc}f1:{f1}")
#         # with open("aaa.txt",'a',encoding='utf-8') as file:
#         #     file.write("text:"+str(aerfa)+",image:"+str(beita)+",mm:"+str(sigema)+"acc:"+str(acc)+"f1:"+str(f1)+"\n")
#         if f1 > best_f12:
#             best_f12 = f1
#             best_acc2 = acc
#             best_pre2 = pre
#             best_rec2 = rec
#             best_parameters2 = {'text': aerfa, 'image': beita, 'mm': sigema}
#         if f1 > best_f1:
#             best_acc = acc
#             best_pre = pre
#             best_rec = rec
#             best_f1 = f1
#             best_parameters = {'text': aerfa, 'image': beita, 'mm': sigema}
#     # Y.append(best_f12)
#     # print(f"best_acc:{best_acc2}")
#     # print(f"precision:{best_pre2}")
#     # print(f"recall:{best_rec2}")
#     # print(f"best_f1:{best_f12}")
#     # print(f"best paramaters{best_parameters2}")
#     # print("--------------------------------")
# print(f"best_acc:{best_acc}")
# print(f"precision:{best_pre}")
# print(f"recall:{best_rec}")
# print(f"best_f1:{best_f1}")
# print(f"best paramaters{best_parameters}")

path = 'clip_cub_'

arr1 = np.load('tau_npy/' + path + 'i2t_256_val.npy', allow_pickle=True)[::-1][:, ::-1][:, :, ::-1]
arr2 = np.load('tau_npy/' + path + 't2i_256_val.npy', allow_pickle=True)[::-1][:, ::-1][:, :, ::-1]

t = 0
tmp = None
x = np.zeros((10, 10, 10))
lamdas = [0.0] * 10
argmax = 0
for i in range(10):
    for j in range(10):
        for k in range(10):
            if t <= arr2[i, j, k]['result']['r_sum']:
                t = arr2[i, j, k]['result']['r_sum']
                tmp = (arr2[i, j, k], i, j, k)
                argmax = i
            if lamdas[i] <= arr2[i, j, k]['result']['r_sum']:
                lamdas[i] = arr2[i, j, k]['result']['r_sum']
print(tmp)
print(lamdas)
arr2 = arr1
for i in range(10):
    for j in range(10):
        for k in range(10):
            x[i, j, k] = arr2[i, j, k]['result']['r_sum']
X = [i / 10 for i in range(1, 11) for _ in range(10)]
Y = [i / 10 for _ in range(10) for i in range(1, 11)]
Z = [a['result']['r_sum'] for b in arr2[argmax] for a in b ]
# Z = np.round(Z, decimals=2)

# 设置全局字体大小（可选）
plt.rcParams.update({'font.size': 24})  # 全局字体大小
if True:
    data = lamdas
    # 生成x轴数据（索引从0到9）
    cx = [(i + 1) / 10 for i in range(len(lamdas))]

    # 设置图片清晰度
    plt.rcParams['figure.dpi'] = 300


    # 绘制折线图
    plt.figure(figsize=(10, 6))
    plt.plot(cx, data, marker='o', linestyle='-', color='#00A1FF', linewidth=2, markersize=6)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # 添加标题和标签
    plt.title('lambda')
    plt.xlabel('value')
    plt.ylabel('Rsum')

    # 调整x轴标签角度
    plt.xticks(cx, rotation=45, ha='right')

    # 移除左、右、上三边的边框
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 可选：保存图表为图片
    plt.savefig('weight_experiment/' + path + 'lambda_val.png', bbox_inches='tight')
x = np.array(x)[argmax]
x = np.round(x, decimals=2)
x = x.flatten()
# print(x)
# print(argmax)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
print(X.shape)
print(Y.shape)
print(Z.shape)

xi = np.linspace(min(X), max(X), 100)
yi = np.linspace(min(Y), max(Y), 100)
XI, YI = np.meshgrid(xi, yi)

# 使用griddata进行插值
ZI = griddata((X, Y), Z, (XI, YI), method='linear')

# 创建画布和主坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 1. 构建 contour_levels
contour_levels = np.concatenate([
    np.sort(x)[::10],
    [np.nanmax(x)]
])
contour_levels = np.unique(np.sort(contour_levels))
print("原始 levels:", contour_levels)

contour_levels = np.concatenate([
    [np.nanmin(x)],
    (contour_levels[:-1] + contour_levels[1:]) / 2,
    [np.nanmax(x)]
])
contour_levels = np.unique(np.sort(contour_levels))
print("中间插值后 levels:", contour_levels)

# 2. 设置 colormap 和归一化
cmap = plt.cm.coolwarm
norm = BoundaryNorm(contour_levels, ncolors=cmap.N)

# 3. 绘制等高线填色图
cf = ax.contourf(XI, YI, ZI, levels=contour_levels, cmap=cmap, norm=norm)

# 4. 添加与主图等高的 colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 调整pad控制左右距离
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label('Rsum')
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# 5. 添加黑色轮廓线
CS = ax.contour(XI, YI, ZI, levels=contour_levels, colors='k', linewidths=0.5)

# 6. 绘制原始采样点
ax.scatter(X, Y, c='red', s=20)

# 7. 坐标轴设置
ax.set_xlabel(r'$\tau_{I2T}$')
ax.set_ylabel(r'$\tau_{T2T}$')
ax.set_aspect('equal', adjustable='box')

# 8. 整图整体往上移
fig.subplots_adjust(top=0.95, bottom=0.05)

# 9. 显示 + 保存
plt.show()
fig.savefig('weight_experiment/' + path + 'i2t_val.png', dpi=300, bbox_inches='tight')

# ====================================================================================================================
arr1 = np.load('tau_npy/' + path + 'i2t_256_val.npy', allow_pickle=True)[::-1][:, ::-1][:, :, ::-1]
arr2 = np.load('tau_npy/' + path + 't2i_256_val.npy', allow_pickle=True)[::-1][:, ::-1][:, :, ::-1]

t = 0
tmp = None
x = np.zeros((10, 10, 10))
lamdas = [0.0] * 10
argmax = 0
for i in range(10):
    for j in range(10):
        for k in range(10):
            if t <= arr2[i, j, k]['result']['r_sum']:
                t = arr2[i, j, k]['result']['r_sum']
                tmp = (arr2[i, j, k], i, j, k)
                argmax = i
            if lamdas[i] <= arr2[i, j, k]['result']['r_sum']:
                lamdas[i] = arr2[i, j, k]['result']['r_sum']
print(tmp)
print(lamdas)
# arr2 = arr1
for i in range(10):
    for j in range(10):
        for k in range(10):
            x[i, j, k] = arr2[i, j, k]['result']['r_sum']
X = [i / 10 for i in range(1, 11) for _ in range(10)]
Y = [i / 10 for _ in range(10) for i in range(1, 11)]
Z = [a['result']['r_sum'] for b in arr2[argmax] for a in b ]

x = np.array(x)[argmax]
x = np.round(x, decimals=2)
x = x.flatten()
# print(x)
# print(argmax)

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
print(X.shape)
print(Y.shape)
print(Z.shape)

xi = np.linspace(min(X), max(X), 100)
yi = np.linspace(min(Y), max(Y), 100)
XI, YI = np.meshgrid(xi, yi)

# 使用griddata进行插值
ZI = griddata((X, Y), Z, (XI, YI), method='linear')

# 创建画布和主坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 1. 构建 contour_levels
contour_levels = np.concatenate([
    np.sort(x)[::10],
    [np.nanmax(x)]
])
contour_levels = np.unique(np.sort(contour_levels))
print("原始 levels:", contour_levels)

contour_levels = np.concatenate([
    [np.nanmin(x)],
    (contour_levels[:-1] + contour_levels[1:]) / 2,
    [np.nanmax(x)]
])
contour_levels = np.unique(np.sort(contour_levels))
print("中间插值后 levels:", contour_levels)

# 2. 设置 colormap 和归一化
cmap = plt.cm.coolwarm
norm = BoundaryNorm(contour_levels, ncolors=cmap.N)

# 3. 绘制等高线填色图
cf = ax.contourf(XI, YI, ZI, levels=contour_levels, cmap=cmap, norm=norm)

# 4. 添加与主图等高的 colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)  # 调整pad控制左右距离
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label('Rsum')
cbar.ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

# 5. 添加黑色轮廓线
CS = ax.contour(XI, YI, ZI, levels=contour_levels, colors='k', linewidths=0.5)

# 6. 绘制原始采样点
ax.scatter(X, Y, c='red', s=20)

# 7. 坐标轴设置
ax.set_xlabel(r'$\tau_{T2I}$')
ax.set_ylabel(r'$\tau_{I2I}$')
ax.set_aspect('equal', adjustable='box')

# 8. 整图整体往上移
fig.subplots_adjust(top=0.95, bottom=0.05)

# 9. 显示 + 保存
plt.show()
fig.savefig('weight_experiment/' + path + 't2i_val.png', dpi=300, bbox_inches='tight')