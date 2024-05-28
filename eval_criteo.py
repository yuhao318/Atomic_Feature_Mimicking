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

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    target = ['Label']

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )

    # # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    # for feat in sparse_features:
    #     lbe = LabelEncoder()
    #     data[feat] = lbe.fit_transform(data[feat])
    # mms = MinMaxScaler(feature_range=(0, 1))
    # data[dense_features] = mms.fit_transform(data[dense_features])

    # data.to_csv("data/criteo/processed_train.csv", index=False)

    # data = pd.read_csv('data/criteo/processed_train.csv')
    data = pd.read_pickle('data/criteo/processed_train.pkl')
    # print(data.dtypes)
    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique(), 16)
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]
    # for s in data.columns:
    #     print(s, data[s].nunique())
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)
    # print(feature_names)
    # 3.generate input data for model

    train, test = train_test_split(data, test_size=0.1, random_state=2020)
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
    #                task='binary', dnn_hidden_units=(400, 400, 400), l2_reg_embedding=1e-5,  device=device)
    # print(model)
    # checkpoint = torch.load("checkpoint/deepfm_criteo_10_400_400_400/tf_transfer_deepfm_400*3_e10.pt")
    # msg = model.load_state_dict(checkpoint)
    # print(msg)
    # checkpoint = torch.load("checkpoint/deepfm_criteo_16_400_400_400/deepfm_criteo_16_400_400_400_0.7964.pt")['model']
    # msg = model.load_state_dict(checkpoint)
    # print(msg)

    # model = DeepFM_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, tags = [False, True, True], dims = [0, 64, 64], 
    #                task='binary', dnn_hidden_units=(400,400,400), l2_reg_embedding=1e-5, device=device)
    # print(model)

    # model = DeepFM_EMB_Ada_Act(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns, num_sparse_features = 26, emb_input_dims = 2, emb_output_dims = 16, tags = [False, True, True], dims = [0, 64, 64], 
    #                task='binary', dnn_hidden_units=(400,400,400), l2_reg_embedding=1e-5, device=device)
    # print(model)

 


    pred_ans = model.predict(test_model_input, 10000)
    print("")
    # print(history)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))