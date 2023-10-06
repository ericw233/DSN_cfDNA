import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from sklearn.metrics import roc_auc_score
from copy import deepcopy

from model import DSN
from train_randomsplit import DSNwithRS
from load_data import load_data_1D_impute
from functions import SIMSE, DiffLoss, MSE, CustomDataset


### set parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
torch.cuda.empty_cache()

data_dir="/mnt/binf/eric/Mercury_Aug2023_new/Feature_all_Aug2023_DomainKAG9v0.csv"
input_size=2600
feature_type="Griffin"
output_path="/mnt/binf/eric/DSN_Sep2023_results/Test_0929/"
code_size=256

############ quick train with random split 3:1
DSN_randomsplit = DSNwithRS(input_size=input_size, code_size=code_size, num_class=2, num_domain=2)        
DSN_randomsplit.data_loader(data_dir=data_dir, input_size=input_size, feature_type=feature_type, seed=98)
DSN_randomsplit.train_with_randomsplit(output_path=output_path)

print("====================  Complete training with random split  ===================")

### load data
data_full, X_train_source_tensor, y_train_source_tensor, d_train_source_tensor, X_train_target_tensor, y_train_target_tensor, d_train_target_tensor, X_test_tensor, y_test_tensor, d_test_tensor, X_all_tensor, _, _, _ = load_data_1D_impute(data_dir, input_size, feature_type) 

data_idonly = data_full[["SampleID","Train_Group"]]
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)
d_test_tensor = d_test_tensor.to(device)

# train_source_dataset = CustomDataset(X_train_source_tensor, y_train_source_tensor)
# train_source_dataloader = data.DataLoader(dataset=train_source_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)

# train_target_dataset = CustomDataset(X_train_target_tensor, y_train_target_tensor)  
# train_target_dataloader = data.DataLoader(dataset=train_target_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)

# test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
# test_dataloader = data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)

### define optimizer and loss functions
DSN_model = DSN(input_size=input_size, code_size=code_size, num_class=2, num_domain=2)
DSN_model.to(device)

### visualize dimensions of model outputs
from torchinfo import summary
tensor_size = (1, 1, X_train_source_tensor.size(2))
print(X_train_source_tensor.size(2))

arg_dict={"mode":'target', "scheme":'all', "p":0.0}
# summary(DSN_model, input_size=tensor_size, **arg_dict)
# summary(DSN_model, input_size=tensor_size)

########################
learning_rate = 1e-4
optimizer = optim.Adam(DSN_model.parameters(), lr=learning_rate, weight_decay=1e-6)

def exp_lr_scheduler(optimizer, step, init_lr=learning_rate, lr_decay_step=1000, step_decay_weight=0.95):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))
    if step % lr_decay_step == 0:
        print(f"current lr is {current_lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


loss_classification = nn.BCELoss()
loss_domainsimilarity = nn.BCELoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()


### training processes
# active_domain_loss_step = 1000
num_epochs = 500
batch_size = 256
alpha = 0.05
beta = 0.05
gamma = 0.25

# len_dataloader = min(len(train_source_dataloader), len(train_target_dataloader))
# dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)
num_iterations_target = (X_train_target_tensor.size(0) // batch_size) + 1
num_iterations_source = (X_train_source_tensor.size(0) // batch_size) + 1

current_step = 0
max_auc=float(0.0)
patience=50
best_model=None
epochs_without_improvement=0

for epoch in range(num_epochs):
        
    # Mini-batch training
    seed = 42 + epoch
    source_shuffled_indices = torch.randperm(X_train_source_tensor.size(0))
    X_train_source_tensor = X_train_source_tensor[source_shuffled_indices].to(device)
    y_train_source_tensor = y_train_source_tensor[source_shuffled_indices].to(device)
    d_train_source_tensor = d_train_source_tensor[source_shuffled_indices].to(device)
    
    target_shuffled_indices = torch.randperm(X_train_target_tensor.size(0))
    X_train_target_tensor = X_train_target_tensor[target_shuffled_indices].to(device)
    y_train_target_tensor = y_train_target_tensor[target_shuffled_indices].to(device)
    d_train_target_tensor = d_train_target_tensor[target_shuffled_indices].to(device)
    
    for batch_start in range(0, min(X_train_target_tensor.size(0), X_train_source_tensor.size(0)), batch_size):
        batch_end = batch_start + batch_size
        ith = batch_start // batch_size
        p = (ith + epoch * num_iterations_target) / (num_epochs * num_iterations_target)
        alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        
        ### target domain training
        DSN_model.train()
        optimizer.zero_grad()
        batch_X_target = X_train_target_tensor[batch_start:batch_end]
        batch_y_target = y_train_target_tensor[batch_start:batch_end]
        batch_d_target = d_train_target_tensor[batch_start:batch_end]
        
        batch_X_target = batch_X_target.to(device)
        batch_y_target = batch_y_target.to(device)
        batch_d_target = batch_d_target.to(device)

        output_target = DSN_model(batch_X_target, mode = "target", scheme = "all", p = p)
        private_code_target, shared_code_target, domain_label_target, recons_code_target = output_target
        
        loss_target = 0
        loss_dann_target =  loss_domainsimilarity(domain_label_target, batch_d_target)
        loss_diff_target = loss_diff(private_code_target, shared_code_target)
        loss_mse_target = loss_recon1(recons_code_target, batch_X_target)
        loss_simse_target = loss_recon1(recons_code_target, batch_X_target)
        loss_target = gamma*loss_dann_target + beta*loss_diff_target + alpha*loss_mse_target + alpha*loss_simse_target
        
        loss_target.backward()
        optimizer.step()

        ### source domain training
        optimizer.zero_grad()
        batch_X_source = X_train_source_tensor[batch_start:batch_end]
        batch_y_source = y_train_source_tensor[batch_start:batch_end]
        batch_d_source = d_train_source_tensor[batch_start:batch_end]
        
        batch_X_source = batch_X_source.to(device)
        batch_y_source = batch_y_source.to(device)
        batch_d_source = batch_d_source.to(device)
        
        output_source = DSN_model(batch_X_source, mode = "source", scheme = "all", p = p)
        private_code_source, shared_code_source, domain_label_source, class_label_source, recons_code_source = output_source
        
        loss_source = 0
        loss_dann_source = loss_domainsimilarity(domain_label_source, batch_d_source)
        loss_class_source = loss_classification(class_label_source, batch_y_source)        
        loss_diff_source = loss_diff(private_code_source, shared_code_source)
        loss_mse_source = loss_recon1(recons_code_source, batch_X_source)
        loss_simse_source = loss_recon1(recons_code_source, batch_X_source)
        loss_source = loss_class_source + gamma*loss_dann_source + beta*loss_diff_source + alpha*loss_mse_source + alpha*loss_simse_source
        
        loss_source.backward()
        optimizer.step()
        
        current_step += 1   # step is for active_domain_loss_step setup      
     
        ### print results of last batch
        # auc_source = roc_auc_score(batch_y_source.to("cpu"), class_label_source.detach().cpu().numpy())
        # print(f"--------   Epoch: {epoch+1}/{num_epochs}, i: {ith}   --------")
        # print(f"Train source total loss: {loss_source.item():.4f}, Train source task loss: {loss_class_source.item():.4f}, ")
        # # print(f"Train target total loss: {loss_target.item():.4f}, Train target domain loss: {loss_dann_target.item():.4f}, ")
        # print("===================================================")
        # print(f"Train source auc: {auc_source.item():.4f}")
        # print("--------------------------------------")     
        
    ### evaluate train-source, train-target, and test for each epoch 
    with torch.no_grad():
        DSN_model.eval()
    
        output_source_full = DSN_model(X_train_source_tensor, mode = "source", scheme = "all", p = 0.1)
        _, _, _, class_label_source_full, _ = output_source_full
        class_label_source_full = class_label_source_full.detach().cpu().numpy()
        
        output_test_full = DSN_model(X_test_tensor, mode = "source", scheme = "all", p = 0.1)
        _, _, _, class_label_test_full, _ = output_test_full
        class_label_test_full = class_label_test_full.detach().cpu().numpy()
        
        auc_source = roc_auc_score(y_train_source_tensor.to("cpu"), class_label_source_full)
        auc_test = roc_auc_score(y_test_tensor.to("cpu"), class_label_test_full)    
            
        print(f"--------   Epoch: {epoch+1}/{num_epochs}   --------")
        print(f"Train source total loss: {loss_source.item():.4f}, Train source task loss: {loss_class_source.item():.4f}, ")
        print(f"Train target total loss: {loss_target.item():.4f}, Train target domain loss: {loss_dann_target.item():.4f}, ")
        print("===================================================")
        print(f"Train source auc: {auc_source.item():.4f}, Test auc: {auc_test.item():.4f}, ")
        print("--------------------------------------")
    
        # Early stopping check
        if auc_source >= max_auc:
            max_auc = auc_test
            best_model = deepcopy(DSN_model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered! No improvement in {patience} epochs.")
                print(f"Best test AUC: {max_auc}")
                break
        
    # torch.save(DSN_model.state_dict(), output_path + 'DSN_epoch_' + str(epoch) + '.pth')
torch.cuda.empty_cache()

DSN_model.train()
DSN_model.load_state_dict(best_model)
torch.save(DSN_model.state_dict(), f"{output_path}/DSN_model_{feature_type}.pth")


with torch.no_grad():
    DSN_model.eval()
    X_all_tensor = X_all_tensor.to("cpu")
    DSN_model.to("cpu")
    output_all = DSN_model(X_all_tensor, mode = "source", scheme = "all", p = 0.1)
    
    _, _, _, class_label_all, _ = output_all
    # class_label_all = class_label_all.detach().cpu()
    data_idonly["DSN_score"] = class_label_all
    data_idonly.to_csv(f"{output_path}/{feature_type}_score.csv", index=False)
                
print("Training completed")
        


