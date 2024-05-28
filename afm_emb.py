from tqdm import tqdm
import os

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, feat_dimension, out_dim):
        super(Net, self).__init__()
        self.emb = nn.Embedding(feat_dimension, out_dim)
    def forward(self, x):
        x = self.emb(x)
        return x    

class CompressEMB_Net(nn.Module):
    def __init__(self, feat_dimension, hidden_dim, out_dim):
        super(CompressEMB_Net, self).__init__()
        self.emb = nn.Embedding(feat_dimension, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, out_dim)
    def forward(self, x):
        x = self.emb(x)
        x = self.layer1(x)
        return x    

feat_dimension, hidden_dim, out_dim = 128, 5, 10
model = Net(feat_dimension, out_dim)
compress_model = CompressEMB_Net(feat_dimension, hidden_dim, out_dim)


# Step 1:
## Train model with training set D.


# Steq 2:
## Given a pre-trained model, collect embedding output in training set D. Then calculate the compression weight

def cal_output_sum_square(fc_output, fc_square, fc_sum, fc_count):
    tensor_i = fc_output
    fc_sum = torch.sum(tensor_i, dim=0)
    fc_square = tensor_i.T @ tensor_i
    fc_count += tensor_i.shape[0]
    if fc_square is None:
        fc_sum = fc_sum
        fc_square = fc_square
    else:
        fc_sum += fc_sum
        fc_square += fc_square

def cal_save_uvb(fc_square, fc_sum, fc_count, save_path= "test"):
    avg_tensor_i = fc_sum.reshape(-1, 1) / fc_count_list[i]
    cov_tensor_i = fc_square / fc_count_list[i] -  avg_tensor_i @ avg_tensor_i.T
    # u,v, avg = cal_cov(cov_tensor_i, avg_tensor_i)
    u,s,v = torch.linalg.svd(cov_tensor_i, full_matrices=False)
    u_name = save_path  + "_u.pt"
    torch.save(u, u_name)
    v_name = save_path   + "_v.pt"
    torch.save(v, v_name)
    avg_name = save_path  + "_avg.pt"
    torch.save(avg_tensor_i, avg_name)

def step2():
    device = 'cuda:0'
    fc_square = None
    fc_sum = None
    fc_count = 0
    with torch.no_grad():
        for ind, x in tqdm(enumerate(train_data_loader)):
            
            # Assume x in Batch_size*input_dim and y in Batch_size*output_dim
            x = x.to(device).float()
            y_pred = model(x)

            cal_output_sum_square(y_pred, fc_square, fc_sum, fc_count)


        save_floader = "path/to/save/floader/"
        save_name = "save_name"
        if not os.path.exists(save_floader):
            os.makedirs(save_floader)
        save_path = save_floader + save_name
        cal_save_uvb(fc_square_list, fc_sum_list, fc_count_list,  save_path )

step2()

# Steq 3:
## Given the compression weight, then merge model embedding

def step3():
    checkpoint = model.state_dict()
    emb_name = 'emb'
    com_emb_name = 'layer1'

    root_path = "path/to/save/floader/"
    root_name = "save_name"

    u_name = root_path +  root_name + "_u.pt"
    v_name = root_path + root_name + "_v.pt"
    avg_name = root_path + root_name + "_avg.pt"

    w_name = emb_name + ".weight"

    u = torch.load(u_name , map_location='cpu')
    v = torch.load(v_name , map_location='cpu')
    avg = torch.load(avg_name , map_location='cpu')
    w = checkpoint[w_name].cpu()


    new_e = (w - avg) @ u[:, :hidden_dim]
    checkpoint[w_name] = new_e

    new_checkpoint[com_emb_name + '.weight'] = u[:, :hidden_dim]
    new_checkpoint[com_emb_name + '.bias'] = avg
step3()