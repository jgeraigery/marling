# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch  
import torch.nn as nn  
import torch.optim as optim  

# Random Network Distillation (RND) Target Network  
class RNDTargetNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, device="cuda"):
        super(RNDTargetNetwork, self).__init__()

        self.device = device

        self.layers = nn.ModuleList()
        layer_dims = [input_dim] + hidden_layers + [output_dim]

        for i in range(len(layer_dims) - 2):  
            layer = nn.Linear(layer_dims[i], layer_dims[i+1])  
            nn.init.xavier_uniform_(layer.weight)  
            self.layers.append(layer)
            self.layers.append(nn.ReLU())  
          
        layer = nn.Linear(layer_dims[-2], layer_dims[-1])  
        nn.init.xavier_uniform_(layer.weight)  
        self.layers.append(layer)

        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:  
            x = layer(x) 
        return x 
    
    # Get the parameters of the target network
    def get_params(self, device="cpu"):
        # Move parameters to the specified device and return as a list
        params = [param.to(device) for param in self.parameters()]
        return params

# Random Network Distillation (RND) Predictor Network 
class RNDPredictorNetwork(nn.Module):  
    def __init__(self, input_dim, output_dim, hidden_layers, learning_rate=1e-2, device="cuda"):  
        super(RNDPredictorNetwork, self).__init__()  
  
        self.device = device  
  
        self.layers = nn.ModuleList()  
        layer_dims = [input_dim] + hidden_layers + [output_dim]  
  
        for i in range(len(layer_dims) - 2):  
            layer = nn.Linear(layer_dims[i], layer_dims[i+1])  
            nn.init.xavier_uniform_(layer.weight)  
            self.layers.append(layer)  
            self.layers.append(nn.ReLU())  
          
        layer = nn.Linear(layer_dims[-2], layer_dims[-1])  
        nn.init.xavier_uniform_(layer.weight)  
        self.layers.append(layer)
  
        self.to(self.device) 
  
    def forward(self, x):  
        for layer in self.layers:  
            x = layer(x) 
        return x 
    
    def get_params(self, device="cpu"):
        # Move parameters to the specified device and return as a list
        params = [param.to(device) for param in self.parameters()]
        return params