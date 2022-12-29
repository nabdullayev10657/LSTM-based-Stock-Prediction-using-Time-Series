import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import Create_Dataset, train, test, calculate_metrics
import matplotlib.pyplot as plt
from model import LSTM

# PARAMETERS
batch_size = 64
input_dim = 1 
hidden_size = 50
num_layers = 3
epochs = 300

df = pd.read_csv('GOOG.csv')
df1 = df.reset_index()['High']

#PREPROCESSING
scalar = MinMaxScaler(feature_range=(0,1))
df1 = scalar.fit_transform(np.array(df1).reshape(-1,1))

#PLOTTING DATASET
plt.figure(figsize = (24, 14))
plt.subplot(2, 2, 1)
plt.title("Total Dataset Visualization")
plt.plot(df['Date'], df['High'])

# SPLITTING DATASET
training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

#PLOTTING TRAINING & TEST DIVISION
plt.subplot(2, 2, 2)
plt.title("Train & Test Division")
plt.plot(df[0:training_size]['High'])
plt.plot(df[training_size:len(df)]['High'])
plt.legend(['Train', 'Test'])

# DATASET CREATION
train_dataset = Create_Dataset(train_data) 
test_dataset = Create_Dataset(test_data) 

# SPLITTING BATCHES WITH DATALOADERS
train_dataloader = DataLoader(train_dataset,batch_size,drop_last=True)
test_dataloader = DataLoader(test_dataset,batch_size , drop_last=True)

# MODEL, LOSS FUNCTION, OPTIMIZER INITIALIZATION
model = LSTM(input_dim , hidden_size , num_layers,batch_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# TRAINING PHASE
print("Training starting...")

for epoch in range(epochs):
    print(f"Results for {epoch} epoch")
    with open('train_test_results.txt', 'a') as f:
        f.write(f"Results for {epoch} epoch \n")
    train(train_dataloader,model,loss_fn,optimizer,batch_size)
    test(test_dataloader,model,loss_fn,batch_size)

# LOSS CALCULATION, PREDICTION VALIDATION
train_mse, y_train_arr, pred_train_arr = calculate_metrics(train_dataloader,model,loss_fn,batch_size,scalar)
test_mse, y_test_arr, pred_test_arr = calculate_metrics(test_dataloader,model,loss_fn,batch_size,scalar)

print(f"Final Train MSE loss {train_mse}")
with open('train_test_results.txt', 'a') as f:
        f.write(f"\n\nFinal Train MSE loss {train_mse}")

print(f"Final Test MSE loss {test_mse}")
with open('train_test_results.txt', 'a') as f:
        f.write(f"Final Test MSE loss {test_mse}")

# PLOTTING PERFORMANCE
plt.subplot(2, 2, 3)
plt.title("Train Performance")
plt.plot(np.array(y_train_arr))
plt.plot(np.array(pred_train_arr))
plt.legend(['Actual Price', 'Train Prediction'])
plt.savefig('visualization.png')

plt.subplot(2, 2, 4)
plt.title("Test Performance")
plt.plot(np.array(y_test_arr))
plt.plot(np.array(pred_test_arr))
plt.legend(['Actual Price', 'Test Prediction'])
plt.savefig('visualization.png')