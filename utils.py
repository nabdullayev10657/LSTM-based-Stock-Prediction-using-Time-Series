import torch
from torch.utils.data import Dataset
from sklearn.metrics import mean_squared_error
import math

class Create_Dataset(Dataset):
    def __init__(self, data, seq_len = 50):
        self.data = torch.from_numpy(data).float().view(-1)
        self.seq_len = seq_len
    
    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len], self.data[idx + self.seq_len]

    def __len__(self):
        return len(self.data)-self.seq_len-1



def train(dataloader, model, loss_function, optimizer, batch_size):
    h_n, c_n = model.init()
    model.train()
    for batch_num, item in enumerate(dataloader):
        x, y = item
        out, h_n, c_n = model(x.reshape(50,batch_size,1),h_n,c_n)
        loss = loss_function(out.reshape(batch_size) , y)
        h_n = h_n.detach()
        c_n = c_n.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_num == len(dataloader)-1:
            loss = (loss.item()/y.sum().item())*100
            print(f"Train loss for this epoch: {loss:>7f} ")
            with open('train_test_results.txt', 'a') as f:
                f.write(f"Train loss for this epoch: {loss:>7f} \n")

def test(dataloader, model, loss_function, batch_size):
    h_n, c_n = model.init()
    model.eval()
    for batch_num, item in enumerate(dataloader):
        x, y = item
        out, h_n, c_n = model(x.reshape(50,batch_size,1),h_n,c_n)
        loss = loss_function(out.reshape(batch_size) , y)
       
        if batch_num == len(dataloader)-1:
            loss = (loss.item()/y.sum().item())*100 
            print(f"Test loss for this epoch: {loss:>7f} ")
            with open('train_test_results.txt', 'a') as f:
                f.write(f"Test loss for this epoch: {loss:>7f} \n\n")

def calculate_metrics(data_loader, model, loss_function, batch_size, scalar):
    pred_arr = []
    y_arr = []
    with torch.no_grad():
        h_n, c_n = model.init()
        for _, item in enumerate(data_loader):
            x, y = item
            x = x.view(50,64,1)
            pred = model(x,h_n,c_n)[0]
            pred = scalar.inverse_transform(pred.detach().numpy()).reshape(-1)
            y = scalar.inverse_transform(y.detach().numpy().reshape(1,-1)).reshape(-1)
            pred_arr = pred_arr + list(pred)
            y_arr = y_arr + list(y)
        return math.sqrt(mean_squared_error(y_arr,pred_arr)), y_arr, pred_arr
