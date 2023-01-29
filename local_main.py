import matplotlib.pyplot as plt 
import numpy as np 
from local_model import MLP
from device import proj_device
import pickle 
import torch 
from tqdm import tqdm
device = proj_device.device

def plot_log(log):
    for k in log.keys():
        plt.figure()
        plt.plot(log[k])
        plt.title(str(k))
    plt.show()

def train(hidden, num_layers, num_epochs, batch_size, test_batch_size = 2000, lr = 1e-3):
    log = {'train_loss': np.zeros(num_epochs), 'test_loss': np.zeros(num_epochs)}
    with open('train_data_dict_local.pickle', 'rb') as handle:
        train_data_dict = pickle.load(handle)
    with open('test_data_dict_local.pickle', 'rb') as handle:
        test_data_dict = pickle.load(handle)
    X_train, Y_train = train_data_dict['X'].to(device), train_data_dict['Y'].to(device)
    num_f = X_train.shape[1]
    X_train, Y_train = torch.split(X_train, batch_size), torch.split(Y_train, batch_size)
    num_train_batches = len(Y_train)
    X_test, Y_test = test_data_dict['X'].to(device), test_data_dict['Y'].to(device)
    X_test, Y_test = torch.split(X_test, test_batch_size), torch.split(Y_test, test_batch_size)
    num_test_batches = len(Y_test)
    model = MLP(num_f, hidden, num_layers)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(num_epochs):
        model.train()
        running_loss = 0.0 
        for b in tqdm(range(num_train_batches)):
            optimizer.zero_grad()
            x, y = X_train[b], Y_train[b]
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        with torch.no_grad():
            log['train_loss'][i] = running_loss / num_train_batches
            test_mean_loss = 0.0
            model.eval()
            for b in range(num_test_batches):
                x, y = X_test[b], Y_test[b]
                out = model(x)
                test_loss = criterion(out, y)
                test_mean_loss += test_loss.item()
            log['test_loss'][i] = (test_mean_loss / num_test_batches)
            print('test loss =', log['test_loss'][i])
    return log 

hidden = 256
num_layers = 4
batch_size = 1024
num_epochs = 150 
log = train(hidden, num_layers, num_epochs, batch_size,)
plot_log(log)

