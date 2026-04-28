# Copyright (c) 2026 Electronic Arts Inc. All rights reserved. See LICENSE for details.

import torch
import torch.nn as nn  
import torch.optim as optim  

# Network for the state abstraction used for the intrinsic bonus term (psi)
class AutoEncoderStateRepresentor(nn.Module):  
    def __init__(self, input_dim, hidden_dim, encoding_dim=64):  
        super(AutoEncoderStateRepresentor, self).__init__()  
        self.encoder = nn.Sequential(  
            nn.Linear(input_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, encoding_dim)  
        )  
        self.decoder = nn.Sequential(  
            nn.Linear(encoding_dim, hidden_dim),  
            nn.ReLU(),  
            nn.Linear(hidden_dim, input_dim)  
        )  
  
    def forward(self, x):  
        encoded = self.encoder(x)  
        decoded = self.decoder(encoded)  
        return decoded  
  
    def encode(self, x):  
        return self.encoder(x)

    
# Intrinsic bonus term (B_w)
class IntrinsicBonus:  
    def __init__(self, autoencoder, B_max, lambd, lr, device="cuda"):  
        self.autoencoder = autoencoder  
        self.device = device
        self.omega = {}
        self.B_max = B_max
        self.lambd = lambd
        self.criterion = nn.MSELoss()  
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=lr)  
  
    def train_autoencoder(self, state):  
        self.optimizer.zero_grad()  
        decoded = self.autoencoder(state)  
        loss = self.criterion(decoded, state)  
        loss.backward()  
        self.optimizer.step
  
    def update_omega(self, state):  
        psi = self.autoencoder.encode(state).detach().numpy().tobytes()  
        if psi not in self.omega:  
            self.omega[psi] = 0  
        self.omega[psi] += 1  
  
    def get_pseudo_count(self, state):  
        psi = self.autoencoder.encode(state).detach().numpy().tobytes()  
        if psi not in self.omega:  
            return 0
        pseudo_count = ((self.lambd / self.B_max) ** 2) + self.omega[psi]  
        return pseudo_count
  
    def get_intrinsic_bonus(self, state):  
        N_w = self.get_pseudo_count(state)  
        B_w = self.lambd / (N_w ** 0.5)  
        return B_w