from logging import getLogger
import math
from typing import Optional
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.parallel import parallelize_module
from .colwise_embedding_bag import ColwiseEmbeddingBag, xFormerEmbeddingBag
import einx
from einops import rearrange, repeat, reduce, pack, unpack
from torch import nn, einsum
from .conv1d import CausalDepthwiseConv1d
logger = getLogger()

class UltraMemory(nn.Module):
    def __init__(self,
                 input_dim: int, 
                 output_dim: int, 
                 value_dim:int,
                 key_dim:int,
                 key_num:int=512,
                 head:int=2,
                 knn:int=16,
                 vitual_num:int=4,
                 is_casual: bool = True,
                 head_core:int=2
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vitual_num = vitual_num
        self.is_casual = is_casual
        self.key_dim = key_dim
        self.key_num = key_num
        self.head=head
        self.knn=knn
        self.value_dim=value_dim
        #设置value和key，还有E矩阵
        self.key=nn.Parameter(torch.empty(2 * self.head*self.key_num*2, self.key_dim // 2))
        self.valuegroup=nn.Parameter(torch.empty(self.vitual_num, value_dim ,value_dim//head_core))
        self.out=nn.Linear(value_dim,output_dim)
        self.core=torch.randn(head, head)
        self.core1=torch.randn(head, head)
        self.Querynet = Querynet(input_dim, key_dim, head, vitual_num)
        self.Valueblock=Valueblock(value_dim=self.value_dim,value_size=key_num*key_num)

    def init_parameters(self):
        #初始化key和value
        nn.init.xavier_uniform_(self.valuegroup)
        #初始化core矩阵
        nn.init.xavier_uniform_(self.core)
    
    def get_indices(self, query: torch.Tensor,knn,core,score_r,score_c):
        #assert query.dim() == 2 and query.size(1) == self.key_dim
        device=query.device
        core=core.to(device)
        
        #svd矩阵获取
        U, S, T = torch.linalg.svd(core, full_matrices=False)
        u = U[:, 0]  # 取U的第一列 -> 对应最大奇异值的左奇异向量
        t = T[0, :]
        u=u.to(device)
        t=t.to(device)
        query = query.view(-1, self.head, self.key_dim//2)
        bs = len(query) 
        half = self.key_dim // 2
        # keys : (heads, 2, n_keys, half)
        # keys1 : (heads, n_keys, half)
        keys = self.key.view(self.head, 2, -1, half) #head 代表多头
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        scores_r = torch.einsum(
            "blh, lkh->blk", query, keys1
        ).to(device)  # (bs , heads, n_keys ** 0.5)
        scores_c= torch.einsum(
            "blh, lkh->blk", query, keys2
        ).to(device)  # (bs , heads, n_keys ** 0.5)
        
        m_score_r=torch.einsum("bhk,h->bk", scores_r, u).to(device)
        m_score_c=torch.einsum("bhk,h->bk", scores_c, t).to(device)
        
        indices_r = torch.topk(m_score_r, knn, dim=-1).indices.to(device)#.expand(bs,self.knn,self.knn)
        indices_c = torch.topk(m_score_c, knn, dim=-1).indices.to(device)#.expand(bs,self.knn,self.knn) #(bs,knn)
        r=indices_r.unsqueeze(-1).expand(bs,self.knn,self.knn).to(device)
        c=indices_c.unsqueeze(-1).expand(bs,self.knn,self.knn).to(device)
    
        
        f_scores_r = torch.gather(input=scores_r,dim=2,index=indices_r.unsqueeze(1).expand(-1, scores_r.size(1), -1)).to(device)
        f_scores_r=f_scores_r.permute(0, 2, 1).to(device)
        f_scores_c = torch.gather(input=scores_c,dim=2,index=indices_c.unsqueeze(1).expand(-1, scores_c.size(1), -1)).to(device)
        #print(f_scores_r.shape,f_scores_c.shape,self.core.shape)
        m_score=torch.einsum("bkh,hh,bhd->bkd", f_scores_r, core, f_scores_c).to(device)
        m_indices = r*self.key_num*(self.vitual_num**0.5) + c  # (bs, knn, knn)
        
        m_score=m_score.flatten(start_dim=-2, end_dim=-1)  #(b,k*k)
        m_indices = m_indices.flatten(start_dim=-2, end_dim=-1)
        f_score,final_indices = torch.topk(m_score, knn+4, dim=-1) 
        
        f_indices= torch.gather(m_indices,dim=1,index=final_indices.long())
        # (bs, knn, 1)
        labels,f_indices = self.classify_indices(f_indices, self.key_num,self.vitual_num**0.5)

        labels[:,-4:] = torch.tensor([0, 1, 2, 3])#这里是为了放防止有一个矩阵没有在前向传播中用到，强制让每个虚拟块都使用

        return f_score,f_indices,labels #(b,k)

    def classify_indices(self,indices, a: int,k:int) -> torch.Tensor:
        if not torch.all((indices >= 0) & (indices < (k*a)**2)):
            raise RuntimeError("模型输入索引越界")

        block_rows = (indices //(k*a)) // a
        block_cols = (indices % (k*a)) // a 
        
        # 计算块编号和相对索引，合并为一个表达式
        block_number = block_rows * (k) + block_cols
        relative_index = ((indices // a) % a) * a + (indices % a)
       
        return block_number,relative_index
    

    ''' def classify_indices_1(self,indices, a: int) -> torch.Tensor:
        if not torch.all((indices >= 0) & (indices < a**2)):
            raise RuntimeError("模型输入索引越界")

        rows = indices// a
        cols = indices% a
        
        mid = a // 2  # 划分中点
        labels = torch.full((len(indices),len(indices[0])), -1, dtype=torch.long)
        
        # 创建掩码
        upper_mask = rows < mid
        lower_mask = rows >= mid
        left_mask = cols < mid
        right_mask = cols >= mid
        # 分配象限标签
        labels[upper_mask & left_mask] = 0    # 左上
        labels[upper_mask & right_mask] = 1   # 右上
        labels[lower_mask & left_mask] = 2     # 左下
        labels[lower_mask & right_mask] = 3   # 右下
        indices[upper_mask & right_mask]-=mid
          # 右上
        indices[lower_mask & left_mask] =(rows[lower_mask & left_mask]-mid)*a+cols[lower_mask & left_mask]    # 左下
        indices[lower_mask & right_mask] =(rows[lower_mask & right_mask]-mid)*a+cols[lower_mask & right_mask]-mid
          # 右下
        return labels '''
    
    
    def forward(self,input):
        B,T,D= input.shape
        device=input.device
        assert D == self.input_dim, f"Input dimension {D} does not match expected {self.input_dim}"
        #x: [B, T, D]
        #input = input.view(-1, self.input_dim)
        query=self.Querynet(input)
        half = self.key_dim // 2
        query = query.view(-1, self.head, self.key_dim//2)
        keys = self.key.view(self.head, 2, -1, half) #head 代表多头
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        scores_r = torch.einsum(
            "blh, lkh->blk", query, keys1
        )  # (bs , heads, n_keys ** 0.5)
        scores_c= torch.einsum(
            "blh, lkh->blk", query, keys2
        )  # (bs , heads, n_keys ** 0.5)
          # [B, num_heads, D//2]

        score_1,index_1,label_1= self.get_indices(query=query, knn=self.knn,core=self.core,score_r=scores_r,score_c=scores_c)  # [B, num_heads, knn]
        score_2,index_2,label_2= self.get_indices(query=query, knn=self.knn,core=self.core1,score_r=scores_r,score_c=scores_c)
        output1 = self.Valueblock(index_1, score_1, self.valuegroup, label_1,1)
        output2 = self.Valueblock(index_2, score_2, self.valuegroup, label_2,0)
        output=torch.cat((output1,output2),dim=-1)
        output= F.dropout(
            output, p=0.1, training=self.training
        )  # [B, num_heads, D//2]
        return self.out(output.reshape(B,T,D//2)) 
        
class Querynet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_heads: int=8, vitual_num:int=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.vitual_num = vitual_num
        #这里用了共享query
        self.querymlp=  nn.Linear(input_dim, num_heads*(output_dim//2), bias=False)
        self.casual=CausalDepthwiseConv1d(in_channels=self.input_dim, out_channels=self.input_dim,kernel_size=3, dilation=1)
    def forward(self,input):
        #input: [B, D]
        #output: [B, D//2]
         # Flatten the input
        bs=len(input)*len(input[0])
        
        output = self.casual(input)
        output = output.reshape(-1,self.input_dim)
        output=self.querymlp(output)
       
        
        assert output.shape == (bs,self.num_heads*(self.output_dim//2)) 
        return output.contiguous().view(bs * self.num_heads, self.output_dim//2)

class Valueblock(nn.Module):
    def __init__(self, value_dim: int, value_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(value_size, value_dim))
        self.value_dim=value_dim
        self.value_size=value_size
        
    def forward(self,indices,scores,W,label,index):
        #score ,label,indices (bs,knn) w (dim,dimout)
        device=W.device
        if index==1:
            weight=self.weight[...,self.value_dim//2:]#后一半
            W=W[...,self.value_dim//2:,:]
        else:
            weight=self.weight[...,:self.value_dim//2]#前一半
            W=W[...,:self.value_dim//2,:]
        b,t = indices.shape
        a,c = W[0].shape
        label=label.to(device)
        indices = indices.view(-1)
        selected_values = torch.index_select(
        weight, 
        dim=0, 
        index=indices.long())  
        selected_values = selected_values.view(b, t, self.value_dim//2)
        expanded_params = W.unsqueeze(0).unsqueeze(0).expand(b,t,-1,-1,-1)
        expanded_params = torch.gather(
                         expanded_params,
                         dim=2,
                         index=label.long().unsqueeze(-1).unsqueeze(-1).expand(b,t,a,c).unsqueeze(2)).squeeze(2)
        expanded_params = expanded_params.squeeze(2)  
        output=torch.einsum("bkd,bkdl,bk->bkl", selected_values, expanded_params,scores) 
        output=torch.sum(output,dim=-2) # (bs, knn, dim) @ (bs, knn) @ (dim, dimout)
        return output
        



