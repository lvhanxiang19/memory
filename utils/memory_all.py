import torch
import torch.nn as nn
import torch.nn.functional as F
import einx
import math
import warnings
from torch import Tensor
from einops import rearrange, reduce, einsum
from functools import partial
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from timm.layers import trunc_normal_


class Memory_value(nn.Module):
    def __init__(self,value_size,value_dim):
        super().__init__()
        self.value_size=value_size
        self.value_dim=value_dim
        self.weight=nn.Parameter(torch.randn(value_size,value_dim))
        
    def forward(self,score,indices,dispatch,n):
        assert indices.dim()==3 
        assert score.dim()==3 
        if dispatch is not None:
            assert indices.dim()==3
            b, e, c = indices.shape
            d = self.weight.size(1)
    
            # 步骤1: 提取知识库向量
            gathered_vectors = self.weight[indices]  # (b, e, c, d)
            
             # 步骤2: 应用得分权重
            weighted_vectors = (gathered_vectors * score.unsqueeze(-1)).view(b,-1,d)  # (b, e, c, d)
            # 步骤3: 按token编号散射重组
            output = torch.zeros(b,n,d, device=self.weight.device)
            output.scatter_add_(
                dim=1,
                index=dispatch.view(b,-1).unsqueeze(-1).expand(-1, -1,d),
                src=weighted_vectors
            )

            
        else:
         score=score.unsqueeze(-1)  # (b,t,knn,1)
         output=self.value[indices]*score
         output=output.sum(dim=2)
        return output

class CausalDepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, dilation_rate=1):
        super().__init__()
        self.in_channels=in_channels
        padding = (kernel_size - 1) * dilation_rate          # 只在左侧填充，保证因果
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,                         # 深度卷积：输出=输入
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation_rate,
            groups=in_channels,
            
        )

    def forward(self, x):
        assert x.dim()==3 and x.size(2)==self.in_channels
        B,T,C=x.shape
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        # 裁掉右侧对称填充
        if self.conv.padding[0] > 0:
            x = x[:, :, :-self.conv.padding[0]]

        x = x.permute(0, 2, 1).reshape(B,T,C)
        return x

class Querynet(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size=3,
                 dilation_rate=1,
                 is_casual=True,
                 bias=True,
                 batchnorm=False,
                 query_dropout=0.0,
                 ):
        super().__init__()
        self.layers=[]
        self.query_dropout=query_dropout
        self.input_dim=input_dim
        self.conv=None
        if is_casual:
           self.conv=CausalDepthwiseConv1d(in_channels=input_dim,kernel_size=kernel_size,dilation_rate=dilation_rate)      
            
        self.layers.append(nn.Linear(input_dim,output_dim,bias=bias))
        if batchnorm:
            self.layers.append(nn.BatchNorm1d(output_dim))
        self.net=nn.Sequential(*self.layers)

    def forward(self,input):
        assert input.dim()==3 and input.size(2)==self.input_dim
        if self.conv:
           input=self.conv(input)
        output=self.net(input)
        output=F.dropout(
            output, p=self.query_dropout, training=self.training
        ) 
        return output


class token_wise_choice(nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 ##router setting
                 key_num=256,
                 key_dim=512,
                 heads=4,
                 knn=32,
                 ##querynet setting
                 is_casual=True,
                 batch_norm=True,
                 query_dropout=0.0,
                 #other
                 **kwargs: Any,):
        assert 0<=query_dropout<=1
        super().__init__()

        self.input_dim=input_dim
        self.output_dim=output_dim
        self.key_num=key_num
        self.key_dim=key_dim
        self.heads=heads
        self.value_size=key_num*key_num
        self.knn=knn
        self.query_dropout=query_dropout
        self.querydim=heads*key_dim

        self.keys = nn.Parameter(
            torch.empty(2 * self.heads * self.key_num, self.key_dim // 2)#这里的key的维度是输入(q)的一半
        )
        self.querynet=Querynet(input_dim=input_dim,output_dim=self.querydim)
    
    def get_indices(self, query, knn,T):
        assert query.dim() == 2 and query.size(1) == self.key_dim
        bs = len(query) // self.heads
        query = query.view(-1, self.heads, self.key_dim)
        half = self.key_dim // 2
        # keys : (heads, 2, n_keys, half)
        # keys1 : (heads, n_keys, half)
        keys = self.keys.view(self.heads, 2, -1, half) #head 代表多头
        keys1 = keys[:, 0, :, :]
        keys2 = keys[:, 1, :, :]
        n_keys = len(keys[0][0])

       

        # split query for product quantization
        q1 = query[:, :, :half]  # (bs, heads, half)  #将query一分为二
        q2 = query[:, :, half:]  # (bs, heads, half)

        # compute indices with associated scores
        scores1 = torch.einsum(
            "blh, lkh->blk", q1, keys1
        )  # (bs , heads, n_keys ** 0,5)
        scores2 = torch.einsum(
            "blh, lkh->blk", q2, keys2
        )  # (bs , heads, n_keys ** 0,5)
        
        scores1, indices1 = scores1.topk(knn, dim=2, largest=True)  # (bs, heads, knn)
        scores2, indices2 = scores2.topk(knn, dim=2, largest=True)  # (bs, heads, knn)

        all_scores = (
            scores1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            + scores2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(
            bs, self.heads, -1
        )  # (bs, heads, knn ** 2)
        all_indices = (
            indices1.view(bs, self.heads, knn, 1).expand(bs, self.heads, knn, knn)
            * n_keys
            + indices2.view(bs, self.heads, 1, knn).expand(bs, self.heads, knn, knn)
        ).view(
            bs, self.heads, -1
        )  # (bs, heads, knn ** 2)

        # select overall best scores and indices
        scores, best_indices = torch.topk(
            all_scores, k=knn, dim=2, largest=True, sorted=True
        )  # (bs, heads, knn)
        indices = all_indices.gather(2, best_indices)  # (bs, knn)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, self.heads, knn)
        return scores.view(int(bs/T),T ,self.heads*knn), indices.view(int(bs/T),T,self.heads*knn)
    
    def forward(self,x):
        B,T,C=x.shape
        query=self.querynet(x)
        query=F.dropout(
            query, p=self.query_dropout, training=self.training
        )
        assert query.shape==(B,T,self.querydim)
        query=query.view(B*T*self.heads,self.key_dim)
        score,indices=self.get_indices(query=query,knn=self.knn,T=T)

        return score,indices,None  #这里的dispatch是None

class block_wise_choice(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 ##router setting
                 key_num=256,
                 key_dim=512,
                 heads=4,
                 knn=32,
                 block_rate=2,
                 ##querynet setting
                 is_casual=True,
                 batch_norm=True,
                 query_dropout=0.0,
                 #other
                 **kwargs: Any,):
        assert 0<=query_dropout<=1
        super().__init__()

        self.input_dim=input_dim
        self.output_dim=output_dim
        self.key_num=key_num
        self.key_dim=key_dim
        self.heads=heads
        self.value_size=key_num*key_num
        self.knn=knn
        self.query_dropout=query_dropout

        self.block=knn*block_rate
        self.col=self.value_size//self.block

        self.querydim=heads*key_dim//2

        self.querynet=Querynet(input_dim=input_dim,output_dim=self.querydim)

        self.rowkeys=nn.Parameter(torch.empty(self.block,heads,key_dim//2))  #(block,h,d/2)
        self.colkeys=nn.Parameter(torch.empty(self.col,heads,key_dim//2)) #(col,h,d/2)
    
    def get_indices(self,query):
        # b-batch, 
        # t-sequence len,
        # h-heads,
        # d-keydim ,
        # c-token per block ,
        # e-block number
        # l-col number
        assert query.dim()==4 and query.size(3)==self.key_dim//2
        B,H,T,C=query.shape 
        device=query.device

        k=(T*self.knn//self.block) #token per block

        query=query.view(B,self.heads,T,self.key_dim//2) #(b,h,t,d)

        base = torch.arange(self.block, dtype=torch.int32)  # 形状 (e,)
        #块索引
        indices = base.view(1, 1, 1, self.block).expand(B, self.heads, T, self.block).to(device)  #(b,h,t,e)
        #块相关矩阵，相当于expert
        score=torch.einsum(
            "bhtd,ehd->bhte",query,self.rowkeys)
        #由于每一行相对于列是独立的，所以只要取出每一行最优score和indices再与行score与indices相加即可
        score_col,indices_col=torch.einsum( 
            "bhtd,chd->bhtc",query,self.colkeys).topk(1,dim=-1,largest=True) #(b,h,t,1)
        
        #(b,h,t,e) #每个token对应的block中的最大分数，索引
        indices=indices*self.col+indices_col #(b,h,t,e) 
        score+=score_col               #(b,h,t,e)
        score=score.view(B,H,-1,T)
        indices=indices.view(B,H,-1,T) #(b,h,e,t)
        _,dispatch=score.topk(k=k,dim=-1,largest=True)  #(b,h,e,k)
        score=score.gather(dim=-1,index=dispatch)
        indices=indices.gather(dim=-1,index=dispatch) #(b,h,e,k)
        
        return score.view(B,self.block,-1),indices.view(B,self.block,-1),dispatch.view(B, self.block, -1) #(b,e,h*c)
    
    def forward(self,x):
        B,T,C=x.shape
        query=self.querynet(x)
        query=F.dropout(
            query, p=self.query_dropout, training=self.training
        )  # (bs * heads, k_dim)
        assert query.shape==(B,T,self.querydim)
        query=query.view(B,self.heads,T,self.key_dim//2)
        score,indices,dispatch=self.get_indices(query=query)

        return score,indices,dispatch



class memory(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            ##router/process setting
            key_num=256,
            key_dim=512,
            value_dim=-1,
            heads=4,
            knn=32,
            block_rate=8,
            router_type='block_wise',
            ##querynet setting
            is_casual=True,
            batch_norm=False,
            query_dropout=0.0,
            ##other setting
            value_proj=True,
            value_proj_bias=True,
            swilu_proj=True,
            swilu_bias=True
    ):
        super().__init__()
        assert 0 <= query_dropout <= 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.key_num = key_num
        self.key_dim = key_dim  
        self.value_size = key_num * key_num
        self.heads = heads
        self.knn = knn
        if value_dim ==-1:
            self.value_dim = key_dim
        else:
            self.value_dim = value_dim
        self.block_rate = block_rate
        self.router_typr = router_type
        self.swilu_projection = swilu_proj
        #初始化value_proj
        self.value_proj=None
        if value_proj:
            self.value_proj = nn.Linear(input_dim,output_dim,bias = value_proj_bias)

        #初始化router
        self.router=None
        if router_type == 'token_wise':
            self.router = token_wise_choice(
                input_dim=input_dim,
                output_dim=output_dim,
                key_num=key_num,
                key_dim=key_dim,
                heads=heads,
                knn=knn,
                is_casual=is_casual,
                batch_norm=batch_norm,
                query_dropout=query_dropout
            )
        elif router_type == 'block_wise':
            self.router = block_wise_choice(
                input_dim=input_dim,
                output_dim=output_dim,
                key_num=key_num,
                key_dim=key_dim,
                heads=heads,
                knn=knn,
                block_rate=block_rate,
                is_casual=is_casual,
                batch_norm=batch_norm,
                query_dropout=query_dropout
            )
        assert self.router is not None, f"Router type {router_type} is not supported."

        #初始化value
        self.value=Memory_value(
            value_size=self.value_size,
            value_dim=output_dim
        )
        
        #初始化swilu_projection
        self.swilu_projection = None
        if swilu_proj:
            self.swilu_projection = nn.Linear(self.input_dim, self.output_dim, bias=swilu_bias)
    
    def init_weights(self):
        if self.value_proj:
            nn.init.normal_(self.value_proj.weight, mean=0, std=self.output_dim**-0.5)
            if self.value_proj.bias is not None:
                nn.init.constant_(self.value_proj.bias, 0)
        if self.swilu_projection:
            nn.init.normal_(self.swilu_projection.weight, mean=0, std=self.output_dim**-0.5)
            if self.swilu_projection.bias is not None:
                nn.init.constant_(self.swilu_projection.bias, 0)
        nn.init.normal_(self.value.weight, mean=0, std=self.value_size**-0.5)
    def forward(self,x):
            assert x.dim() == 3 and x.size(2) == self.input_dim
            B, T, C = x.shape
            #获取combine和dispatch
            score, indices,dispatch = self.router(x)
            assert score.dim() == 3 
            assert indices.dim() == 3 
            
            #获取value
            output = self.value(score=score, indices=indices, dispatch=dispatch,n=T)
            if self.value_proj:
                output = self.value_proj(output.view(B, T, -1))
            if self.swilu_projection:
                output=output*F.silu(self.swilu_projection(x))
            return output

        