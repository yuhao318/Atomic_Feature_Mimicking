import time

import numpy as np
import torch
import torch.nn as nn

from ..layers import FM, DNN


# class FM(nn.Module):
#     """FM part"""
 
#     def __init__(self, latent_dim, fea_num):
#         """
#         latent_dim: 各个离散特征隐向量的维度  8
#         input_shape: 这个最后离散特征embedding之后的拼接和dense拼接的总特征个数   221
#         """
#         super(FM, self).__init__()
 
#         self.latent_dim = latent_dim
#         # 定义三个矩阵， 一个是全局偏置，一个是一阶权重矩阵， 一个是二阶交叉矩阵，注意这里的参数由于是可学习参数，需要用nn.Parameter进行定义
#         self.w0 = nn.Parameter(torch.zeros([1, ]))   #1,
#         self.w1 = nn.Parameter(torch.rand([fea_num, 1]))   #221*1
#         self.w2 = nn.Parameter(torch.rand([fea_num, latent_dim]))   #221*8
 
#     def forward(self, inputs):  #32*221
#         # 一阶交叉
#         first_order = self.w0 + torch.mm(inputs, self.w1)  # (samples_num, 1)   w1=221*1    fin 32*1
#         # 二阶交叉  这个用FM的最终化简公式
#         second_order = 1 / 2 * torch.sum(
#             torch.pow(torch.mm(inputs, self.w2), 2) - torch.mm(torch.pow(inputs, 2), torch.pow(self.w2, 2)),
#             dim=1,
#             keepdim=True
#         )  # (samples_num, 1)  32*8     行求和后 32*1
 
#         return first_order + second_order

# class Dnn(nn.Module):
#     """Dnn part"""
 
#     def __init__(self, hidden_units, dropout=0.):
#         """
#         hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
#         dropout = 0.
#         """
#         super(Dnn, self).__init__()
 
#         self.dnn_network = nn.ModuleList(
#             [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
#         self.dropout = nn.Dropout(dropout)
 
#     def forward(self, x):
#         for linear in self.dnn_network:
#             x = linear(x)
#             x = F.relu(x)
#         x = self.dropout(x)
#         return x

class DeepFM(nn.Module):
    def __init__(self, feature_columns, dnn_hidden_units, device='cpu', dnn_dropout=0.,):
        """
        DeepFM:
        :param feature_columns: 特征信息， 这个传入的是fea_cols
        :param hidden_units: 隐藏单元个数， 一个列表的形式， 列表的长度代表层数， 每个元素代表每一层神经元个数
        """
        super(DeepFM, self).__init__()
        self.sparse_feature_cols, self.dense_feature_cols = feature_columns
 
        # embedding
        self.embed_layers = nn.ModuleDict({
            feat['name']: nn.Embedding(num_embeddings=feat['vocabulary_size'], embedding_dim=feat['embedding_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
 
        # 这里要注意Pytorch的linear和tf的dense的不同之处， 前者的linear需要输入特征和输出特征维度， 而传入的hidden_units的第一个是第一层隐藏的神经单元个数，这里需要加个输入维度
        self.fea_num = len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
            'embedding_dim']  #13+26*8   221
        # dnn_hidden_units.insert(0, self.fea_num)  #[221, 128, 64, 32]
 
        self.fm = FM()
        # self.dnn_network = Dnn(hidden_units, dnn_dropout)
        print(self.fea_num, dnn_hidden_units)
        self.dnn = DNN(inputs_dim=self.fea_num, hidden_units=dnn_hidden_units, dropout_rate=dnn_dropout, device=device)

        self.nn_final_linear = nn.Linear(dnn_hidden_units[-1], 1)   #32*1
 
    def forward(self, x):
        sparse_inputs, dense_inputs  = x[:, :len(self.sparse_feature_cols)], x[:, len(self.sparse_feature_cols):]
        sparse_inputs = sparse_inputs.long()  # 转成long类型才能作为nn.embedding的输入
        # sparse_embeds = [self.embed_layers[i](sparse_inputs[:, i]) for i in
        #                  range(sparse_inputs.shape[1])]
        # self.embed_layers = nn.ModuleDict({
        #     feat['name']: nn.Embedding(num_embeddings=feat['vocabulary_size'], embedding_dim=feat['embedding_dim'])
        #     for i, feat in enumerate(self.sparse_feature_cols)
        # })
        sparse_embeds = [self.embed_layers[feat['name']](sparse_inputs[:, i]) for i, feat in enumerate(self.sparse_feature_cols)]
        # for i in sparse_embeds:
        #     print(i.shape)

        # sparse_embeds = torch.cat(sparse_embeds, dim=-1)  #32*208
        sparse_embeds = torch.stack(sparse_embeds, dim=1)  #32*208

    
        # print(sparse_embeds.shape)
        wide_outputs = self.fm(sparse_embeds)  #32*1
        # deep
        # print(wide_outputs.shape)
        x = torch.cat([sparse_embeds.flatten(1,2) , dense_inputs], dim=-1)#32*211
        # print(x.shape)
        deep_outputs = self.dnn(x)   #32*1
        # print(deep_outputs.shape)

        outputs = torch.sigmoid(self.nn_final_linear(torch.add(wide_outputs, deep_outputs)))
        # 模型的最后输出
        # outputs = F.sigmoid(torch.add(wide_outputs, deep_outputs))  #32*1
 
        return outputs


