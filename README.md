**PROJECT OVERVIEW**

In last years, high quality and fast GPUs have been great bridge for advanced implementations of different architectures in Deep Learning (in general, Artificial Intelligence). These great innovations and advanced applications on Big Data have also brought many promises in future prediction and stock market movements. But still, today's current research on stock movement has not shown itself in very precise manner and sometimes, we observe that predicting stocks is quite challenging. Considering this great attention and global research on this field, this project is for predicting stock prices of Google LLC based on their last 5 years of stock movements with LSTM (Long Short Term Memory) model. Here, I have tried to predict prices with 50 past days of time series data.

All in all, I believe that this project (small research) can contribute to global research in this field to some extend and be helpful for others.

**REQUIREMENTS**

In order to be able to successfully use this code, you should first install some libraries from terminal:
```
pip install torch
pip install numpy
pip install -U scikit-learn scipy matplotlib
```

Depending on version of python or facing error cases, you can try the same lines with ``` pip3 ```.

**CLONING PROECT**

For cloning project, you should execute this line:
```
git clone https://github.com/nabdullayev10657/LSTM-based-Stock-Prediction-using-Time-Series.git
```

**RUNNING THE CODE**

After making sure that everything is installed, for running project, you should type:

```
python train.py
```

**RESULTS**

Customized LSTM model has been trained on ```epoches=300```, ```batch_size=64```. As the sequence length, I have chosen ```seq_len=50```. 
For looking end results, we might want to see ```train_test_results.txt``` file to see train and test loss respectively after each epoch. Also, you can see MSE error at the end.

PREDICTIONS

Visualization graphs show total stock movement, train&test data division, and most importantly, how our model learns from data and performs in test data. As can be seen, after some time, our predictions are getting closer and closer to actual price in time series data:

![visualization](https://user-images.githubusercontent.com/83968119/209979204-bc6b1e54-a7a1-4ec0-a750-0e31c5a86fa8.png)

**DATASET**

Dataset link for 5 Years of stock prices of Google: https://finance.yahoo.com/quote/GOOG/history?period1=1514505600&period2=1672272000&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
