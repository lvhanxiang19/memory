import torch
import timm
import torch.nn.functional as F
from ..models import timm_model_register
def init_memory(mem_dim,
mem_k_dim,
mem_k_num,
mem_head,
mem_layer,mem_knn,num_class):
    checkpoint = torch.load('/home/ipad_3d/jet/jet/moejet/weights/vit_tiny_patch16_224.augreg_in21k_ft_in1k.pth', map_location='cpu')

# 检查文件类型（可能是完整模型或纯state_dict）
    if 'model' in checkpoint:  # 若为完整检查点（包含结构和训练参数）
      teacher = checkpoint['model']
    else:                      # 若为纯权重文件
      teacher = checkpoint
    teacher.pop('head.weight')
    teacher.pop('head.bias')
    keys=teacher.keys()
    print(keys)


    student=timm.create_model(model_name='vit_tiny_mem_custom',num_classes=num_class,
    pretrained=False,
    #memory args
    mem_dim=mem_dim,
    mem_k_dim=mem_k_dim,
    mem_k_num=mem_k_num,
    mem_head=mem_head,
    mem_layer=mem_layer,
    mem_knn=mem_knn)

    keys_s=student.state_dict()
    student=keys_s
    keys_s=keys_s.keys()
##稀疏模型的不同参数
    diff_s=student.copy()
##密集模型的不同参数
    diff_t=teacher.copy()

##将所有结构一致的模块进行初始化

    for key in keys_s:
       if key in keys:
        student[key]=teacher[key]
        diff_s.pop(key)
        diff_t.pop(key)
    ##删除相同的block,提取不同的模块


##下面先初始化value，目前的思路是将ffn中的两个矩阵拼接，然后将所有不同的层的拼接矩阵再拼接，之后通过线性插值将value初始化
    value_pre=None
    for num in mem_layer:
       t1=teacher[f'blocks.{num}.mlp.fc1.weight']
       t2=teacher[f'blocks.{num}.mlp.fc2.weight']
       if value_pre==None:
          value_pre=torch.cat((t1,t2.T),dim=0)
       else:
          temp=torch.cat((t1,t2.T),dim=0)
          value_pre=torch.cat((temp,value_pre),dim=0)


##通过线性插值变换维度
    value_pre=F.interpolate(
    value_pre.unsqueeze(0).unsqueeze(0),                
    size=(mem_k_num**2,value_pre.size(1)),            
    scale_factor=None,    
    mode='bilinear',      
    align_corners=False    
)
    value_pre=value_pre[0][0]
    for i in mem_layer:
      if f'blocks.{i}.memory.values.weight' in student.keys():
        student[f'blocks.{i}.memory.values.weight']=value_pre
    
##下面初始化每一层的keys
    for i in mem_layer:
     t_w_k=f'blocks.{i}.attn.qkv.weight'
     s_w_k=f'blocks.{i}.memory.keys'
     dk=teacher[t_w_k].size(0)//3
   
     q,k,v=torch.split(teacher[t_w_k],dk,dim=0)
     key_m=student[s_w_k]
     temp=F.interpolate(
    k.unsqueeze(0).unsqueeze(0),                
    size=student[s_w_k].size(),            
    scale_factor=None,    
    mode='bilinear',      
    align_corners=False)   
    student[s_w_k]=temp[0][0]
    
    print(student.keys())
    torch.save(student,f"/home/ipad_3d/jet/jet/moejet/weights/vit_tiny_mem_value_{mem_k_num**2}.pth")
    return f"/home/ipad_3d/jet/jet/moejet/weights/vit_tiny_mem_value_{mem_k_num**2}.pth"

   
    