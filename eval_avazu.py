# -*- coding: utf-8 -*-
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

if __name__ == "__main__":
    # data = pd.read_csv("data/criteo/train.csv")

    sparse_features = ["hour","C1","banner_pos","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21"]

    target = ['click']

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )

    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # data.to_csv("data/criteo/processed_train.csv", index=False)

    data = pd.read_csv('data/avazu/processed_train.csv')
    print(data.dtypes)
    # 2.count #unique features for each sparse field,and record dense feature field name

    # fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),4)
    #                           for feat in sparse_features]
    # fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),8)
    #                           for feat in sparse_features]

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(),50)
                              for feat in sparse_features]

    # for s in data.columns:
    #     print(s, data[s].nunique())
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    # print(feature_names)
    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2020)
    print(f'train.shape = ',train.shape)
    print(f'test.shape = ',test.shape)


    train_model_input = {name: train[name] for name in feature_names}

    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'
    # device = 'cpu'

    # model = DeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/deepfm_avazu_50_2000_2000_2000/deepfm_avazu_50_2000_2000_2000_0.7917.pt")
    # msg = model.load_state_dict(checkpoint['model'])
    # print(msg)
    # model = DeepFM_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # model = DeepFM_EMB_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, num_sparse_features = 22, emb_input_dims = 8, emb_output_dims = 50, tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)

    ttd_feature_columns = {'ttd_rank' : 16, 'device_id': [2686408, 100, 150, 180],'device_ip': [6729486, 170, 200, 200]}
    model = DeepFM_TTD(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, ttd_feature_columns= ttd_feature_columns,
                   task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)

    # model = DeepFM_EMB(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, num_sparse_features= len(sparse_features), emb_input_dims = 24, emb_output_dims = 50,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/deepfm_avazu_50_2000_2000_2000/deepfm_avazu_50_2000_2000_2000_0.7917_svd.pt")
    # msg = model.load_state_dict(checkpoint, strict=False)
    # print(msg)

    # model = DeepFM_EMB(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, num_sparse_features= len(sparse_features), emb_input_dims = 4, emb_output_dims = 50,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/deepfm_avazu_50_2000_2000_2000/deepfm_avazu_50_2000_2000_2000_0.7917_svd_4.pt")
    # msg = model.load_state_dict(checkpoint, strict=False)
    # print(msg)

    # model = FiBiNET(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/fibinet_avazu_50_2000_2000_2000/fibinet_avazu_50_2000_2000_2000.pt")
    # msg = model.load_state_dict(checkpoint['model'])
    # print(msg)
    # model = FiBiNET_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)

    # model = FiBiNET_EMB_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                 num_sparse_features = len(sparse_features),  emb_input_dims = 8 , emb_output_dims = 50, 
    #                 tags = [False, True, True], dims = [0, 320, 320], 
    #                 task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)

    # model = AutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/AutoInt_avazu_50_2000_2000_2000/AutoInt_avazu_50_2000_2000_2000_0.7904.pt")
    # msg = model.load_state_dict(checkpoint['model'])
    # print(msg)
    # model = AutoInt_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # model = AutoInt_EMB_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
    #                num_sparse_features = len(sparse_features),  emb_input_dims = 8 , emb_output_dims = 50, 
    #                tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)

    # model = DCN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/DCN_avazu_50_2000_2000_2000/DCN_avazu_50_2000_2000_2000_0.789.pt")
    # msg = model.load_state_dict(checkpoint['model'])
    # print(msg)

    # model = DCN_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # model = DCN_EMB_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
    #                num_sparse_features = len(sparse_features),  emb_input_dims = 8 , emb_output_dims = 50, 
    #                tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)

    # model = NFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/NFM_avazu_50_2000_2000_2000/NFM_avazu_50_2000_2000_2000_0.7864.pt")
    # msg = model.load_state_dict(checkpoint['model'])
    # print(msg)

    # model = NFM_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)

    # print(model)

    # model = NFM_EMB_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, 
    #                num_sparse_features = len(sparse_features),  emb_input_dims = 8 , emb_output_dims = 50, 
    #                tags = [False, True, True], dims = [0, 320, 320], 
    #                task='binary', dnn_hidden_units=(2000,2000,2000), l2_reg_embedding=1e-5, device=device)
    # print(model)

    pred_ans = model.predict(test_model_input, 10000)
    print("")
    # print(history)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

