# -*- coding:utf-8 -*-
"""
Author:
    Wutong Zhang
Reference:
    [1] Huang T, Zhang Z, Zhang J. FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1905.09433, 2019.
"""

import torch
import torch.nn as nn

from .basemodel import BaseModel
from ..inputs import combined_dnn_input, SparseFeat, DenseFeat, VarLenSparseFeat
from ..layers import SENETLayer, BilinearInteraction, DNN, DNN_Ada, DNN_Ada_Act


class FiBiNET_Ada_Act(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, tags = None, dims = None, bilinear_type='interaction',
                 reduction_ratio=3, dnn_hidden_units=(128, 128), l2_reg_linear=1e-5,
                 l2_reg_embedding=1e-5, l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu', gpus=None):
        super(FiBiNET_Ada_Act, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)
        self.linear_feature_columns = linear_feature_columns
        self.dnn_feature_columns = dnn_feature_columns
        self.filed_size = len(self.embedding_dict)
        self.SE = SENETLayer(self.filed_size, reduction_ratio, seed, device)
        self.SE_Bilinear = BilinearInteraction(self.filed_size, self.embedding_size, bilinear_type, seed, device)
        self.Bilinear = BilinearInteraction(self.filed_size, self.embedding_size, bilinear_type, seed, device)
        self.dnn = DNN_Ada_Act(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units, tags = tags, dims = dims,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=False,
                       init_std=init_std, device=device)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        field_size = len(sparse_feature_columns)

        dense_input_dim = sum(map(lambda x: x.dimension, dense_feature_columns))
        embedding_size = sparse_feature_columns[0].embedding_dim
        sparse_input_dim = field_size * (field_size - 1) * embedding_size
        input_dim = 0

        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim

        return input_dim

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        sparse_embedding_input = torch.cat(sparse_embedding_list, dim=1)

        # print(sparse_embedding_input.shape)
        # print(self.SE)
        senet_output = self.SE(sparse_embedding_input)
        # senet_bilinear_out = self.Bilinear(senet_output)
        senet_bilinear_out = self.SE_Bilinear(senet_output)
        bilinear_out = self.Bilinear(sparse_embedding_input)

        linear_logit = self.linear_model(X)
        # temp = torch.split(torch.cat((senet_bilinear_out, bilinear_out), dim=1), 1, dim=1)
        # dnn_input = combined_dnn_input(temp, dense_value_list)
        # dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        # temp1 = torch.cat((senet_bilinear_out, bilinear_out), dim=-1).flatten(-2,-1)
        # dnn_input = torch.cat([temp1, dense_dnn_input], dim=-1)
        temp1 = torch.cat((senet_bilinear_out, bilinear_out), dim=-1).flatten(-2,-1)

        if len(dense_value_list) != 0:
            dense_dnn_input = torch.flatten(
                torch.cat(dense_value_list, dim=-1), start_dim=1)
            dnn_input = torch.cat([temp1, dense_dnn_input], dim=-1)
        
        else:
            dnn_input = torch.cat([temp1], dim=-1)


        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        if len(self.linear_feature_columns) > 0 and len(self.dnn_feature_columns) > 0:  # linear + dnn
            final_logit = linear_logit + dnn_logit
        elif len(self.linear_feature_columns) == 0:
            final_logit = dnn_logit
        elif len(self.dnn_feature_columns) == 0:
            final_logit = linear_logit
        else:
            raise NotImplementedError

        y_pred = self.out(final_logit)

        return y_pred,  final_logit, dnn_output
