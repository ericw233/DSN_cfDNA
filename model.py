import torch
import torch.nn as nn
from functions import ReverseLayerF


class DSN(nn.Module):
    def __init__(self, input_size, code_size, num_class, num_domain):
        super(DSN, self).__init__()
        
        self.input_size = input_size
        ##############################
        ### target feature encoder ###
        self.encoder_target_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        
        self.target_fc_input_size = self._get_fc_input_size(input_size,encoder='target')
        
        self.encoder_target_fc = nn.Sequential(
            nn.Linear(self.target_fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, code_size)
        )
          
        ##############################
        ### source feature encoder ###
        self.encoder_source_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=3, stride=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        
        self.source_fc_input_size = self._get_fc_input_size(input_size,encoder='source')
        
        self.encoder_source_fc = nn.Sequential(
            nn.Linear(self.source_fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, code_size)
        )

        ##############################
        ### shared feature encoder ###
        self.encoder_shared_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.0),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        
        self.shared_fc_input_size = self._get_fc_input_size(input_size, encoder='shared')
        
        self.encoder_shared_fc = nn.Sequential(
            nn.Linear(self.shared_fc_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, code_size)
        )

        self.encoder_shared_classpred = nn.Sequential(
            nn.Linear(code_size, 1),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.encoder_shared_domainpred = nn.Sequential(
            nn.Linear(code_size, 1),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            nn.Sigmoid()
        )

        ##############################
        ### shared decoder ###
        self.decoder_shared_fc = nn.Sequential(
            nn.Linear(code_size, 512),
            nn.ReLU(),
            # nn.Dropout(drop3),
            nn.Linear(512, input_size*4*4)   ### input_size*4 * 4 channels
        )  
        
        self.decoder_shared_conv = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=2, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            # nn.Dropout(drop1_de),
            # nn.MaxPool1d(kernel_size=pool1_de, stride=2),
            
            # nn.Conv1d(in_channels=16, out_channels=16, kernel_size=2, stride=2, bias=None),
            # nn.ReLU(),
            # nn.BatchNorm1d(16),
            # # nn.Dropout(drop2_de),
            # # nn.MaxPool1d(kernel_size=pool1_de, stride=2),
            
            # nn.Upsample(scale_factor=2),
            
            # nn.Conv1d(in_channels=32, out_channels=8, kernel_size=2, stride=2, bias=None),
            # nn.ReLU(),
            # nn.BatchNorm1d(8),
            # nn.Dropout(drop2_de),
            # nn.MaxPool1d(kernel_size=pool2_de, stride=2),
            
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=2, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(1)
            
        )
        
        
    ### function to get the size for fc layers    
    def _get_fc_input_size(self, input_size, encoder):
        dummy_input = torch.randn(1, 1, input_size)
        if encoder == 'target':
            x = self.encoder_target_conv(dummy_input)
        elif encoder == 'source':
            x = self.encoder_source_conv(dummy_input)
        elif encoder == 'shared':
            x = self.encoder_shared_conv(dummy_input)
        
        flattened_size = x.size(1) * x.size(2)
        return flattened_size
    
                
    def forward(self, x, mode="source", scheme="all", p=0.0):
        result = []
        
        ### encoder for target domain
        if mode == "target":
            private_feature = self.encoder_target_conv(x)
            private_feature = private_feature.view(private_feature.size(0),-1)
            private_code = self.encoder_target_fc(private_feature)
        
        ### encoder for source domain
        elif mode == "source":
            private_feature = self.encoder_source_conv(x)
            private_feature = private_feature.view(private_feature.size(0),-1)
            private_code = self.encoder_source_fc(private_feature)
        
        result.append(private_code)
        
        ### encoder for both
        shared_feature = self.encoder_shared_conv(x)
        shared_feature = shared_feature.view(shared_feature.size(0),-1)
        shared_code = self.encoder_shared_fc(shared_feature)
        result.append(shared_code)
        
        reverse_shared_code = ReverseLayerF.apply(shared_code, p)
        domain_output = self.encoder_shared_domainpred(reverse_shared_code)
        result.append(domain_output.squeeze(1))
        
        if mode == "source":
            class_output = self.encoder_shared_classpred(shared_code)
            result.append(class_output.squeeze(1))
        
        ### decoder
        if scheme == "shared":
            union_code = shared_code
        elif scheme == "private":
            union_code = private_code
        elif scheme == "all":
            union_code = shared_code + private_code
        
        recons_fc = self.decoder_shared_fc(union_code)
        recons_fc = recons_fc.view(-1, 4, self.input_size*4)   # reconstruction fc layers export 800 variables 
        recons_code = self.decoder_shared_conv(recons_fc)
        result.append(recons_code)
        
        return result
    
    
