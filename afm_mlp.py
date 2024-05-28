from tqdm import tqdm
import os

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        return x    

class CompressMLP_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super(CompressMLP_Net, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        return x    

input_dim, hidden_dim, out_dim = 128, 32, 128
model = Net(input_dim, out_dim)
compress_model = CompressMLP_Net(input_dim, hidden_dim, out_dim)


# Step 1:
## Train model with training set D.


# Steq 2:
## Given a pre-trained model, collect FC output in training set D. Then calculate the compression weight

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
## Given the compression weight, then merge model weight

def step3():
    checkpoint = model.state_dict()
    conv_name = 'layer1'
    com_conv_name = 'layer2'

    root_path = "path/to/save/floader/"
    root_name = "save_name"

    u_name = root_path +  root_name + "_u.pt"
    v_name = root_path + root_name + "_v.pt"
    avg_name = root_path + root_name + "_avg.pt"

    w_name = conv_name + ".weight"
    b_name = conv_name + ".bias"

    u = torch.load(u_name , map_location='cpu')
    v = torch.load(v_name , map_location='cpu')
    avg = torch.load(avg_name , map_location='cpu')
    w = checkpoint[w_name].cpu()
    b = checkpoint[b_name].cpu()

    new_u = u[:,:hidden_dim]
    new_v = v[:hidden_dim,:]


    after_b = ((torch.eye(new_u.shape[0]) - new_u @ new_v) @ avg).reshape(-1)
    after_w = new_u

    new_w = new_v @ w
    new_b = new_v @ b

    checkpoint[w_name] = new_w
    checkpoint[b_name] = new_b

    checkpoint[com_conv_name + ".weight" ] = after_w
    checkpoint[com_conv_name + ".bias"] = after_b
step3()