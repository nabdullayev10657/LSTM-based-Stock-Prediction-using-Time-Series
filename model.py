import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_dim = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.batch_size = batch_size
        self.fc = nn.Linear(hidden_size, 1) # we want 1 output at the end
    
    def forward(self, X, h_n, c_n):
        res, (h_n, c_n) = self.lstm(X, (h_n, c_n))
        res = self.fc(res[-1])
        return res, h_n, c_n
    
    def predict(self, X):
        h_n = torch.zeros(self.num_of_layers , self.batch_size , self.hidden_size)
        c_n = torch.zeros(self.num_layers , self.batch_size , self.hidden_size)
        res, (h_n, c_n) = self.lstm(X, (h_n, c_n))
        res = self.fc(res[-1])
        return res, h_n, c_n

    def init(self):
        h =  torch.zeros(self.num_layers , self.batch_size , self.hidden_size)
        c =  torch.zeros(self.num_layers , self.batch_size , self.hidden_size)
        return h, c

