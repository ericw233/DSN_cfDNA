import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import inspect

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from copy import deepcopy

from model import DSN
from load_data import load_data_1D_impute   
from pad_and_reshape import pad_and_reshape_1D
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from functions import SIMSE, DiffLoss, MSE, CustomDataset

class DSNwithRS(DSN):
    def __init__(self, input_size=950, code_size=256, num_class=2, num_domain=2):
        super(DSNwithRS, self).__init__(input_size, code_size, num_class, num_domain)
        self.input_size=input_size
    
    ### DSNwithCV has more inputs than the original DSN class, find the proper ones for DSN to initiate the class
    # def _match_params(self, config):
    #     model_config={}
    #     args = inspect.signature(DSN.__init__).parameters
    #     model_keys = [name for name in args if name != 'self']

    #     for key, value in config.items():
    #         if key in model_keys:
    #             model_config[key] = value        
    #     return model_config, model_keys
    
    def data_loader(self, data_dir, input_size, feature_type, seed):
        
        if(input_size != self.input_size):
           raise ValueError("input size of data does not match")
        
        self.feature_type=feature_type
                    
        # Read data from CSV file
        data = pd.read_csv(data_dir)

        # Keep a full dataset without shuffling
        mapping = {'Healthy':0,'Cancer':1}
        
        # Split the data into train and test sets
        train_source_index = data.loc[(data["train"] == "training") & (data["Domain"] != -1)].index
        train_target_index = data.loc[(data["train"] == "training") & (data["Domain"] == 0)].index
        train_source_domain = data.loc[train_source_index,"Domain"]
        
        traintrain_source_index, trainvalid_source_index, _, _ = train_test_split(train_source_index, train_source_domain, test_size = 0.25, stratify = train_source_domain, random_state=seed) 
        traintrain_target_index = [x for x in train_target_index if x not in trainvalid_source_index]

        X_traintrain_source = data.iloc[traintrain_source_index].filter(regex = feature_type, axis=1)
        y_traintrain_source = data.loc[traintrain_source_index,"Train_Group"].replace(mapping)
        d_traintrain_source = data.loc[traintrain_source_index,"Domain"]
        
        X_traintrain_target = data.iloc[traintrain_target_index].filter(regex = feature_type, axis=1)
        y_traintrain_target = data.loc[traintrain_target_index,"Train_Group"].replace(mapping)
        d_traintrain_target = data.loc[traintrain_target_index,"Domain"]
        
        X_trainvalid_source = data.iloc[trainvalid_source_index].filter(regex = feature_type, axis=1)
        self.y_trainvalid_source = data.loc[trainvalid_source_index,"Train_Group"].replace(mapping)
        self.sampleid_trainvalid = data.loc[trainvalid_source_index,"SampleID"]
        
        #### drop constant NA columns based on X_traintrain_source
        na_columns = X_traintrain_source.columns[X_traintrain_source.isna().all()]
        X_traintrain_source_drop = X_traintrain_source.drop(columns = na_columns)
        X_traintrain_target_drop = X_traintrain_target.drop(columns = na_columns)
        X_trainvalid_source_drop = X_trainvalid_source.drop(columns = na_columns)
                
        #### impute variables based on X_traintrain_source_drop
        mean_imputer = SimpleImputer(strategy = 'mean')
        X_traintrain_source_drop_imputed = mean_imputer.fit_transform(X_traintrain_source_drop)
        X_traintrain_target_drop_imputed = mean_imputer.transform(X_traintrain_target_drop)
        X_trainvalid_source_drop_imputed = mean_imputer.transform(X_trainvalid_source_drop)
        
        # Scale the features to a suitable range (e.g., [0, 1])
        scaler = MinMaxScaler()
        X_traintrain_source_scaled = scaler.fit_transform(X_traintrain_source_drop_imputed)
        X_traintrain_target_scaled = scaler.transform(X_traintrain_target_drop_imputed)
        X_trainvalid_source_scaled = scaler.fit_transform(X_trainvalid_source_drop_imputed)

        # Convert the data to PyTorch tensors
        self.X_traintrain_source_tensor = pad_and_reshape_1D(X_traintrain_source_scaled, self.input_size).type(torch.float32)
        self.y_traintrain_source_tensor = torch.tensor(y_traintrain_source.values, dtype=torch.float32)
        self.d_traintrain_source_tensor = torch.tensor(d_traintrain_source.values, dtype=torch.float32)
        
        self.X_traintrain_target_tensor = pad_and_reshape_1D(X_traintrain_target_scaled, self.input_size).type(torch.float32)
        self.y_traintrain_target_tensor = torch.tensor(y_traintrain_target.values, dtype=torch.float32)
        self.d_traintrain_target_tensor = torch.tensor(d_traintrain_target.values, dtype=torch.float32)
        
        self.X_trainvalid_source_tensor = pad_and_reshape_1D(X_trainvalid_source_scaled, self.input_size).type(torch.float32)
        self.y_trainvalid_source_tensor = torch.tensor(self.y_trainvalid_source.values, dtype=torch.float32)
        
    ### resetting weights after each fold of training
    # def weight_reset(self):
    #     for module in self.modules():
    #         if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
    #             module.reset_parameters()
        
    def train_with_randomsplit(self, output_path):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        ########################
        learning_rate = 1e-4
        decay_rate = 1e-6
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

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
        num_iterations_target = (self.X_traintrain_target_tensor.size(0) // batch_size) + 1
        num_iterations_source = (self.X_traintrain_source_tensor.size(0) // batch_size) + 1

        current_step = 0
        max_auc=float(0.0)
        patience=100
        best_model=None
        best_classpred=[]
        epochs_without_improvement=0

        for epoch in range(num_epochs):
                
            # Mini-batch training
            seed = 42 + epoch
            source_shuffled_indices = torch.randperm(self.X_traintrain_source_tensor.size(0))
            X_traintrain_source_tensor = self.X_traintrain_source_tensor[source_shuffled_indices].to(device)
            y_traintrain_source_tensor = self.y_traintrain_source_tensor[source_shuffled_indices].to(device)
            d_traintrain_source_tensor = self.d_traintrain_source_tensor[source_shuffled_indices].to(device)
            
            target_shuffled_indices = torch.randperm(self.X_traintrain_target_tensor.size(0))
            X_traintrain_target_tensor = self.X_traintrain_target_tensor[target_shuffled_indices].to(device)
            y_traintrain_target_tensor = self.y_traintrain_target_tensor[target_shuffled_indices].to(device)
            d_traintrain_target_tensor = self.d_traintrain_target_tensor[target_shuffled_indices].to(device)
            
            X_trainvalid_source_tensor = self.X_trainvalid_source_tensor.to(device)
            
            for batch_start in range(0, min(X_traintrain_target_tensor.size(0), X_traintrain_source_tensor.size(0)), batch_size):
                batch_end = batch_start + batch_size
                ith = batch_start // batch_size
                p = (ith + epoch * num_iterations_target) / (num_epochs * num_iterations_target)
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
                
                ### target domain training
                self.train()
                optimizer.zero_grad()
                batch_X_target = X_traintrain_target_tensor[batch_start:batch_end]
                batch_y_target = y_traintrain_target_tensor[batch_start:batch_end]
                batch_d_target = d_traintrain_target_tensor[batch_start:batch_end]
                
                batch_X_target = batch_X_target.to(device)
                batch_y_target = batch_y_target.to(device)
                batch_d_target = batch_d_target.to(device)

                output_target = self(batch_X_target, mode = "target", scheme = "all", p = p)
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
                batch_X_source = X_traintrain_source_tensor[batch_start:batch_end]
                batch_y_source = y_traintrain_source_tensor[batch_start:batch_end]
                batch_d_source = d_traintrain_source_tensor[batch_start:batch_end]
                
                batch_X_source = batch_X_source.to(device)
                batch_y_source = batch_y_source.to(device)
                batch_d_source = batch_d_source.to(device)
                
                output_source = self(batch_X_source, mode = "source", scheme = "all", p = p)
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
                self.eval()
            
                output_trainvalid = self(X_trainvalid_source_tensor, mode = "source", scheme = "all", p = 0.1)
                _, _, _, class_label_trainvalid, _ = output_trainvalid
                class_label_trainvalid = class_label_trainvalid.detach().cpu().numpy()
                
                auc_source_trainvalid = roc_auc_score(self.y_trainvalid_source_tensor.to("cpu"), class_label_trainvalid)
                   
                print(f"--------   Epoch: {epoch+1}/{num_epochs}   --------")
                print(f"Train source total loss: {loss_source.item():.4f}, Train source task loss: {loss_class_source.item():.4f}, ")
                print(f"Train target total loss: {loss_target.item():.4f}, Train target domain loss: {loss_dann_target.item():.4f}, ")
                print("===================================================")
                print(f"Train-valid source auc: {auc_source_trainvalid.item():.4f}")
                print("--------------------------------------")
            
                # Early stopping check
                if auc_source_trainvalid >= max_auc:
                    max_auc = auc_source_trainvalid
                    best_model = deepcopy(self.state_dict())
                    best_classpred = class_label_trainvalid
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping triggered! No improvement in {patience} epochs.")
                        print(f"Best test AUC: {max_auc}")
                        break
                
        torch.save(self.state_dict(), f"{output_path}/DSN_model_{self.feature_type}_trainvalid.pth")
        torch.cuda.empty_cache()
        
        output_data = {'SampleID':self.sampleid_trainvalid.values, 'y_label':self.y_trainvalid_source.values, 'DSN_score':class_label_trainvalid}
        output_df = pd.DataFrame(output_data)
        output_df.to_csv(f"{output_path}/{self.feature_type}_score_trainvalid.csv", index=False)
        




        