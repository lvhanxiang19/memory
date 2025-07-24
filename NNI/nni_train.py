import nni
import os
import tempfile
import shutil
import subprocess
import re
from moejet.weights.get_weight_vit_tiny_memory import init_memory
# 获取超参数
params = nni.get_next_parameter()
#a=init_memory(mem_dim=192,mem_k_dim=192,mem_head=params['head'],mem_k_num=params['k_num'],mem_knn=params['knn'],mem_layer=params['memory_layer'],num_class=100)
#print(a)
#params['checkpoint']="'"+a+"'"
#print(params.keys())
#print(params)
config = 'vit_memory.py'
# 读取配置文件
with open(('/home/ipad_3d/jet/jet/moejet/NNI/'+config), 'r') as f:
    config_content = f.read()
placeholders = re.findall(r"\{(.*?)\}", config_content) 


# 用超参数替换占位符
config_content = config_content.format(**params)

# 将更新后的配置文件写入临时文件
temp_dir = tempfile.mkdtemp()
temp_config_path = os.path.join(temp_dir, config)
with open(temp_config_path, 'w') as f:
    f.write(config_content)


port = nni.get_sequence_id() + 29500
work_dir = f'./work_dirs/{nni.get_experiment_id()}'
# 使用subprocess.Popen运行训练并实时读取输出
train_command = f'NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 PORT={port} bash ./tools/dist_train.sh {temp_config_path} 4 --work-dir {work_dir}'
print(train_command)
# CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh RetNet/configs/RetNet-base-p16_cifar10.py 4 --work-dir ./work_dirs/RetNetB_cifar10_b32x4_adamw3e-4
process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE, text=True)

max_acc = 0.0  # Initialize maximum accuracy
train_output = ""
while True:
    # 按行读取输出
    line = process.stdout.readline()
    if not line:
        break
    train_output += line

    # # 如果输出包含训练损失，则报告损失
    # match = re.search(r"Epoch\(train\).*loss:\s+([0-9.]+)", line)
    # if match:
    #     loss = float(match.group(1))
    #     nni.report_intermediate_result(loss)

    # 如果输出包含验证准确率，则报告准确率
    acc_match = re.search(r"Epoch\(val\).*accuracy/top1:\s+([0-9.]+)", line)
    if acc_match:
        acc = float(acc_match.group(1))
        nni.report_intermediate_result(acc)
        if acc > max_acc:
            max_acc = acc

# 等待训练进程结束
process.wait()

# 向NNI报告评价指标
nni.report_final_result(max_acc)

# 清理临时文件
shutil.rmtree(temp_dir)
