import nni
import os
import tempfile
import shutil
import subprocess
import re

def parse_output_and_report_intermediate_result(output):
    # 从输出日志中解析训练损失
    loss_regex = r"Epoch\(train\).*loss:\s+([0-9.]+)"
    loss_list = re.findall(loss_regex, output)

    # 使用nni.report_intermediate_result报告训练损失
    for i, loss in enumerate(loss_list):
        nni.report_intermediate_result(loss, "loss")

    # 从输出日志中解析acc和训练损失
    acc_regex = r"Epoch\(val\).*accuracy/top1:\s+([0-9.]+)"
    acc_list = re.findall(acc_regex, output)

    # 使用nni.report_intermediate_result报告验证acc
    for i, acc in enumerate(acc_list):
        nni.report_intermediate_result(acc, "acc")

def get_best_acc(train_output):
    # 从训练日志中解析acc
    acc_regex = r"Epoch\(val\).*accuracy/top1:\s+([0-9.]+)"
    acc_list = re.findall(acc_regex, train_output)

    # 获取最高的acc
    best_acc = max(acc_list)

    return best_acc

def main():
    # 获取超参数
    params = nni.get_next_parameter()

    config = 'RetNet-p16_cifar10_NNI.py'
    # 读取配置文件
    with open(('./RetNet/configs/NNI/'+config), 'r') as f:
        config_content = f.read()

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
    train_command = f'CUDA_VISIBLE_DEVICES=0,1,2,3 PORT={port} ./tools/dist_train.sh {temp_config_path} 4 --work-dir {work_dir}'
    process = subprocess.Popen(train_command, shell=True, stdout=subprocess.PIPE, text=True)

    train_output = ""
    while True:
        # 按行读取输出
        line = process.stdout.readline()
        if not line:
            break
        train_output += line

        # 如果输出包含训练损失，则报告损失
        # loss_match = re.search(r"Epoch\(train\).*loss:\s+([0-9.]+)", line)
        # if loss_match:
        #     loss = float(loss_match.group(1))
        #     nni.report_intermediate_result(loss)

        # 如果输出包含验证准确率，则报告准确率
        acc_match = re.search(r"Epoch\(val\).*accuracy/top1:\s+([0-9.]+)", line)
        if acc_match:
            acc = float(acc_match.group(1))
            nni.report_intermediate_result(acc)

    # 解析训练日志并获取最高的acc
    best_acc = get_best_acc(train_output)

    # 报告最终结果
    nni.report_final_result(best_acc)

    # 等待训练进程结束
    process.wait()

    # 清理临时文件
    shutil.rmtree(temp_dir)

if __name__ == '__main__':
    main()
