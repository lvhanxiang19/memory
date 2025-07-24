import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import glob
import os
import time
# 假设 model 是训练前的初始模型，trained_model 是训练后的模型

def deal(count,name1,name2):
   model=torch.load(name1,map_location='cpu')
   model_final=torch.load(name2,map_location='cpu')
   model=model['state_dict']
   model_final=model_final['state_dict']

   weight1=model['blocks.1.memory.values.weight']
   weight2=model_final['blocks.1.memory.values.weight']
   color_list = ['black', 'white', 'black']
   custom_cmap = LinearSegmentedColormap.from_list('BlackMid', color_list, N=256)
   d=weight2-weight1
   j=0
   for i in range(d.size(0)):
      is_all_zero = (torch.abs(d[i]).sum().item() <=1e-6)
      if not is_all_zero:
         count[i]=count[i]+1
         j=j+1
   
   return j
    


def _get_sorted_epoch_files(target_dir):
        # 获取所有epoch文件并按数字排序（示例：epoch_100.pth > epoch_99.pth）
        files = glob.glob(os.path.join(target_dir, "epoch_*.pth"))
        files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        return files


count=np.zeros(256*256)
num_t=[]
name_temp=None
name1=None
name2=None
while(1):
   time.sleep(1)
   target_dir='/home/ipad_3d/jet/jet/work_dirs/vit_memory_256'
   final_dir='/home/ipad_3d/jet/jet/work_dirs/vit_memory_512/epoch_300.pth'
   files=_get_sorted_epoch_files(target_dir)
   if len(files)<2:
       continue
   name1=files[0]
   name2=files[1]
   print('现在的name1是')
   print(name1)
   print('现在的name2是')
   print(name2)
   print('-----------')
   name_temp=files[0]
   j=deal(count,name1,name2)
   unique_values, counts = np.unique(count, return_counts=True)
   print(unique_values,counts)
# 确定新数组的索引范围（覆盖旧数组最小到最大值）
   min_val = np.min(count)
   max_val = np.max(count)
   arr_new = np.zeros(300)  # 初始化全零数组

# 填充频率到新数组对应的索引位置
   for i in range(len(unique_values)):
       arr_new[int(unique_values[i])]=counts[i]
   num_t.append(j)
   print(num_t)
     # 设置画布大小（宽，高）
   plt.figure(figsize=(12, 4))
   plt.plot(arr_new, color="blue", linewidth=1.5, alpha=0.8)
   plt.title("update time distrubute", fontsize=12)
   plt.xlabel("update count", fontsize=10)
   plt.ylabel("key_num", fontsize=10)
   plt.grid(True, linestyle="--", alpha=0.5) # 确保横坐标范围为数据长度
   plt.savefig('11.jpg')
   plt.close()
# -------------------- 绘制第二个图（300维） --------------------
   plt.figure(figsize=(12, 4))
   plt.plot(num_t, color="blue", linewidth=1.5, alpha=0.8)
   plt.title("selection of keys per epoch", fontsize=12)
   plt.xlabel("epoch", fontsize=10)
   plt.ylabel("number", fontsize=10)
   plt.grid(True, linestyle="--", alpha=0.5)
   plt.savefig('12.jpg')
   plt.close()
   while(name_temp==name1):
       files=_get_sorted_epoch_files(target_dir)
       name_temp=files[0]
       time.sleep(1)

       
   

   
    