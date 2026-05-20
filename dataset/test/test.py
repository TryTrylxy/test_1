import json
import cv2
import numpy as np
import random
from tqdm import tqdm
import os

DATAPATH = '/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_03/dataset/my_f30k/new_test.json'
# ====================== 超参数设置（可自行调整）======================
IMAGE_PATH = "/home/jncsnlp3/SSD2/syy/LexLIP-ICCV23-main/dataset/F30k/f30k_data/1007129816.jpg"       # 输入图片路径
MIN_AREA_RATIO = 0.1          # 蒙版最小面积占比（原图的0.1）
MAX_AREA_RATIO = 0.2          # 蒙版最大面积占比（原图的0.5）
RANDOM_SEED = 123            # 随机种子（None为不固定，设置数值可复现结果）
OUTPUT_PATH = "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/dataset/test/f30k_test_img_mask/"
OUTPUT_PATH1 = "/home/jncsnlp3/SSD2/syy/aaai24_itr_cusa-main_06/dataset/test/f30k_test_img_change/"
os.makedirs(OUTPUT_PATH1, exist_ok=True)
# ====================================================================

def gen_img(data, date_replace):
    img_path = data['image_path']

    # 读取图片（BGR格式）
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"无法读取图片，请检查路径：{IMAGE_PATH}")

    # 获取图片原始尺寸
    h, w = img.shape[:2]
    total_area = h * w

    # 1. 正态分布随机生成蒙版面积（限制在0.1~0.5倍原图）
    mask_area_ratio = np.random.normal(loc=(MIN_AREA_RATIO+MAX_AREA_RATIO)/2, 
                                    scale=(MAX_AREA_RATIO-MIN_AREA_RATIO)/4)
    # 裁剪到0.1~0.5的范围
    mask_area_ratio = np.clip(mask_area_ratio, MIN_AREA_RATIO, MAX_AREA_RATIO)
    mask_area = total_area * mask_area_ratio

    # 2. 正态分布随机生成蒙版长宽比（1:1为中心，避免比例极端）
    aspect_ratio = np.random.normal(loc=1.0, scale=0.3)
    aspect_ratio = np.clip(aspect_ratio, 0.2, 5.0)  # 限制长宽比在0.2~5之间（避免过窄/过高）

    # 计算蒙版的宽和高
    mask_w = np.sqrt(mask_area * aspect_ratio)
    mask_h = np.sqrt(mask_area / aspect_ratio)
    if mask_w > w:
        mask_w = w
        mask_h = w / aspect_ratio
    if mask_h > h:
        mask_w = h * aspect_ratio
        mask_h = h
    mask_w = int(mask_w)
    mask_h = int(mask_h)

    # 3. 正态分布随机生成蒙版左上角坐标（确保不超出图片边界）
    x = int(np.random.normal(loc=(w-mask_w)/2, scale=(w-mask_w)/6))
    y = int(np.random.normal(loc=(h-mask_h)/2, scale=(h-mask_h)/6))
    # 裁剪坐标到有效范围
    x = np.clip(x, 0, w - mask_w)
    y = np.clip(y, 0, h - mask_h)

    # 生成黑色蒙版（将指定区域设为黑色）
    img_masked = img.copy()
    img_masked[y:y+mask_h, x:x+mask_w] = 0  # 0对应BGR的黑色

    # 保存结果
    cv2.imwrite(OUTPUT_PATH + data['image_id'], img_masked)

    replace_img = cv2.imread(date_replace['image_path'])
    h, w = replace_img.shape[:2]
    if h > mask_h:
        rand_y = np.random.randint(0, h - mask_h + 1)
    else:
        rand_y = 0
        mask_h = h
    if w > mask_w:
        rand_x = np.random.randint(0, w - mask_w + 1)
    else:
        rand_x = 0
        mask_w = w
    rand_x = np.random.randint(0, w - mask_w + 1)
    img_masked[y:y+mask_h, x:x+mask_w] = replace_img[rand_y:rand_y+mask_h, rand_x:rand_x+mask_w]
    cv2.imwrite(OUTPUT_PATH1 + data['image_id'], img_masked)

def main():
    # 设置随机种子（可选）
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    with open(DATAPATH, "r", encoding="utf8") as f:
        dataList = json.load(f)
    for i in tqdm(range(0, len(dataList), 5)):
        data = dataList[i]
        gen_img(data, dataList[random.randint(0, len(dataList) - 1)])

main()