import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import json
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import random
from tqdm import tqdm
# 设置 matplotlib 支持英文显示
# plt.rcParams["font.family"] = ["Arial", "sans-serif"]  # 使用 Arial 字体
# plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

def compute_same_tau(data1, data2, tau1=1.0):
    # 加载并处理第一个数据集
    sim1 = F.cosine_similarity(data1.unsqueeze(1), data1.unsqueeze(0), dim=2)
    tmp1 = F.softmax(sim1 / tau1, dim=1)
    target_max = tmp1.max().item()  # 目标最大值

    # 定义二分查找范围
    tau2_min, tau2_max = 0.01, 10.0  # 初始搜索范围，可根据实际情况调整
    max_iter = 30  # 最大迭代次数
    tolerance = 1e-6  # 容差范围

    for i in tqdm(range(max_iter)):
        tau2_mid = (tau2_min + tau2_max) / 2
        
        # 加载并处理第二个数据集
        sim2 = F.cosine_similarity(data2.unsqueeze(1), data2.unsqueeze(0), dim=2)
        tmp2 = F.softmax(sim2 / tau2_mid, dim=1)
        current_max = tmp2.max().item()
        
        # 调整搜索范围
        if abs(current_max - target_max) < tolerance:
            break
        elif current_max < target_max:
            tau2_max = tau2_mid
        else:
            tau2_min = tau2_mid

    # 使用找到的tau2值
    tau2 = tau2_mid
    print(f"找到的tau2值: {tau2}")

    # 最终计算
    data2 = torch.load('tmp1.pth')
    data2 = torch.nn.functional.normalize(data2, p=2, dim=1)
    sim2 = F.cosine_similarity(data2.unsqueeze(1), data2.unsqueeze(0), dim=2)
    tmp2 = F.softmax(sim2 / tau2, dim=1)

    # 拼接结果
    tmp = torch.concat([tmp1.flatten(), tmp2.flatten()])
    print(f"tmp1最大值: {tmp1.max().item()}")
    print(f"tmp2最大值: {tmp2.max().item()}")
    return tmp1, tmp2

def plot_heatmap(tensor, save_path="heatmap.png", vmin=None, vmax=None, title=None):
    """
    绘制tensor的热力图并保存
    
    参数:
    tensor: 4x4的torch tensor
    save_path: 保存图片的路径
    """
    # 将tensor转换为numpy数组以便绘制
    tensor_np = tensor.numpy()
    
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制热力图
    im = ax.imshow(tensor_np, cmap="viridis", vmin=vmin, vmax=vmax)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Similarity distribution', rotation=270, labelpad=20, fontsize=20)  # 颜色条标签
    
    # 设置坐标轴标签和标题（英文）
    ax.set_title(title)
    
    # 设置x轴标签旋转45度并右对齐
    abcd_labels = ['A', 'B', 'C', 'D']
    ax.set_xticks(np.arange(tensor_np.shape[1]))
    ax.set_yticks(np.arange(tensor_np.shape[0]))
    ax.set_xticklabels(abcd_labels, fontsize=25)
    ax.set_yticklabels(abcd_labels, fontsize=25)
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个格子中显示数值
    for i in range(tensor_np.shape[0]):
        for j in range(tensor_np.shape[1]):
            # text = ax.text(j, i, f'{tensor_np[i, j]:.2f}',
            #               ha="center", va="center", color="w")
            text = ax.text(j, i, f'{tensor_np[i, j]:.2f}',
                          ha="center", va="center", 
                          color="w" if tensor_np[i, j] < (vmin + vmax) / 2 else "k", fontsize=20)
            # text = ax.text(j, i, f'{tensor_np[i, j]:.2e}',  # 使用科学计数法，保留2位小数
            #               ha="center", va="center", 
            #               color="w" if (vmax is not None and tensor_np[i, j] < vmax/2) 
            #                        else ("w" if vmax is None and tensor_np[i, j] > tensor_np.max()/2 else "k"))
            # text = ax.text(j, i, f'{tensor_np[i, j]:.2e}',  # 使用科学计数法，保留2位小数
            #               ha="center", va="center", 
            #               color="w")
    
    # 调整布局并保存图片
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存至: {save_path}")

# 示例：生成一个随机的4x4 tensor
if __name__ == "__main__":
    ids = [142668, 27324, 31253, 117361, 77019, 45128, 100251, 113973, 96208, 79988, 112240, 123272, 72137, 49645, 38922, 2753, 19941, 34268, 36896, 72173, 34667, 142192, 27884, 122273, 77195, 10959, 114678, 52549, 132423, 38860, 79226, 13559, 77026, 108170, 50017, 32820, 61825, 467, 6229, 132085, 114267, 93203, 42140, 27838, 115229, 79207, 42282, 122415, 120668, 120405, 71890, 58787, 40235, 56251, 136886, 59133, 108565, 8772, 54112, 142699, 129291, 107090, 129422, 3904, 86262, 85847, 44037, 2966, 50290, 37238, 31522, 118483, 19183, 104601, 28498, 80088, 22015, 127342, 13993, 20048, 67560, 41147, 68798, 85686, 20006, 91910, 136413, 137815, 17520, 87116, 28182, 118752, 32025, 83842, 65331, 113761, 144973, 89964, 98698, 9659, 15817, 85372, 10526, 43903, 31779, 128543, 136863, 8017, 73650, 131529, 9176, 130987, 112296, 88696, 13784, 105481, 130651, 36908, 16907, 79773, 130519, 22913, 111775, 25003, 98208, 85386, 38155, 143101]
    
    # ids = [id // 5 for id in ids]
    # data = np.load('/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_03/dataset/my_f30k/train_unicom.npy', allow_pickle=True).item()
    # data = [data[key] for key in data]
    # data = np.stack(data)
    # print(data.shape)
    # data = torch.tensor(data[ids])
    # torch.save(data, 'tmp.pth')
    # raise

    # with open('/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_03/dataset/my_f30k/new_train.json', "r", encoding="utf8") as f:
    #     data = json.load(f)
    # tmp = ["A surfer in a black wetsuit catches a small wave", "A man surfing in the ocean", "A man sitting on a ledge eating an apple"]
    # for i in range(3):
    #     for j in range(len(data)):
    #         if tmp[i] in data[j]['caption']:
    #             print(j)
    #             break
    # raise

    # with open('/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_03/dataset/my_f30k/new_train.json', "r", encoding="utf8") as f:
    #     data = json.load(f)
    # for i in range(4):
    #     print(data[ids[i]])
    #     print()
    # raise
    # data = [data[t]['caption'] for t in ids]
    # txt_enc_assisant = SentenceTransformer('/home/jncsnlp3/SSD2/syy/huggingface/all-mpnet-base-v2',cache_folder=r"/home/jncsnlp3/SSD2/syy/huggingface/all-mpnet-base-v2")
    # caption_features = txt_enc_assisant.encode(
    #     data, device='cpu', show_progress_bar=False, convert_to_tensor=True)
    # torch.save(caption_features, 'tmp2.pth')
    # raise

    # tau1 = 1.0
    # data1 = torch.load('tmp.pth')
    # data1 = torch.nn.functional.normalize(data1, p=2, dim=1)
    # print(data1.shape)
    # sim1 = util.cos_sim(data1, data1)
    # tmp1 = F.softmax(sim1 / tau1, dim=1)

    # tau2 = 1.0
    # data2 = torch.load('tmp1.pth')
    # data2 = torch.nn.functional.normalize(data2, p=2, dim=1)
    # print(data2.shape)
    # sim2 = util.cos_sim(data2, data2)
    # tmp2 = F.softmax(sim2 / tau2, dim=1)
    # tmp = torch.concat([tmp1.flatten(), tmp2.flatten()])
    # print(tmp.shape)
    data1 = torch.load('tmp.pth')
    data1 = torch.nn.functional.normalize(data1, p=2, dim=1)
    data2 = torch.load('tmp2.pth')
    data2 = torch.nn.functional.normalize(data2, p=2, dim=1)

    tmp1, tmp2 = compute_same_tau(data1, data2, 0.2)
    # tmp1 *= 10
    # tmp2 *= 10
    tmp = torch.concat([tmp1.flatten(), tmp2.flatten()])

    plot_heatmap(tmp1[:4, :4], save_path="heatmap1.png", vmin=min(tmp), vmax=max(tmp))
    plot_heatmap(tmp2[:4, :4], save_path="heatmap2.png", vmin=min(tmp), vmax=max(tmp))
    # plot_heatmap(tmp1[:4, :4], save_path="heatmap1.png")
    # plot_heatmap(tmp2[:4, :4], save_path="heatmap2.png")