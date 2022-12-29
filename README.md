This project is for predicting stock prices of Google LLC. In order to be able to successfully use this code, you should first install some libraries from terminal:

**REQUIREMENTS**
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

![visualization](https://user-images.githubusercontent.com/83968119/209979204-bc6b1e54-a7a1-4ec0-a750-0e31c5a86fa8.png)
