import torch
import numpy as np
import pandas as pd

root = "/home/jncsnlp3/SSD2/syy/CUB"
print(torch.load(root + '/metadata.pth').keys())
data_imgs = torch.load(root + '/imgs_train_val_256x256.pth')
data_meta = torch.load(root + '/metadata.pth')
data_ids_train_val = torch.load(root + '/metadata.pth')['train_val_img_ids']
dataList = []
annotations = pd.read_csv(root + '/res.json', sep='\t', header=None)
caps = np.array(annotations)
for i in range(len(data_ids_train_val)):
    for k in range(len(data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]])):
        data_cap = [data_meta['word_id_to_word'][j] for j in data_meta['img_id_to_encoded_caps'][data_ids_train_val[i]][k] if j != 717]
        tmp = {
            'image_id': data_ids_train_val[i],
            'caption': ' '.join(data_cap),
            'caption_r': caps[i][0],
            'class_id': data_meta['img_id_to_class_id'][data_ids_train_val[i]],
            'img': data_imgs[i]
        }
        dataList.append(tmp)
arg = [(i + t) * 10 + u for i in range(0, len(dataList) // 10 , 10) for t in range(9) for u in range(10)]
dataList = [dataList[t] for t in arg if t < len(dataList)]

print(dataList[1])
print(dataList[23])
print(dataList[1459])