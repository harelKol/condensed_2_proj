from model import MLP 
import pickle 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from device import proj_device
import random
device = proj_device.device

def plot_log(log):
    for k in log.keys():
        plt.figure()
        plt.plot(log[k])
        plt.title(str(k))
    plt.show()
        

def train(hidden, blocks, num_epochs, batch_size, test_batch_size = 100, lr = 1e-4):
    log = {'train_loss': np.zeros(num_epochs), 'test_loss': np.zeros(num_epochs)}
    # MODEL, LOSS, OPTIM#
    model = MLP(hidden, blocks)
    model.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # DATA # 
    with open('train_data_dict.pickle', 'rb') as handle:
        train_data_dict = pickle.load(handle)
    with open('test_data_dict.pickle', 'rb') as handle:
        test_data_dict = pickle.load(handle)
    #Split to Batches # 
    #train 
    X, Y, mask = train_data_dict['X'].to(device), train_data_dict['Y'].to(device), train_data_dict['mask'].to(device)
    X, Y = torch.split(X, batch_size), torch.split(Y, batch_size)
    num_batches = len(Y)
    ind = list(range(num_batches))
    #test 
    X_test, Y_test, mask_test = test_data_dict['X'].to(device), test_data_dict['Y'].to(device), test_data_dict['mask'].to(device)
    X_test, Y_test = torch.split(X_test, test_batch_size), torch.split(Y_test, test_batch_size)
    num_test_batches = len(Y_test)

    A_ratio = mask.shape[1] / torch.sum(mask)
    
    for i in range(num_epochs):
        model.train()
        print('epoch', str(i), 'from', str(num_epochs))
        running_loss = 0.0 
        random.shuffle(ind)
        for b in tqdm(range(num_batches)):
            x = X[ind[b]]
            y = Y[ind[b]]
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out * mask, y * mask)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        with torch.no_grad():
            log['train_loss'][i] = running_loss / num_batches
            log['train_loss'][i] *= A_ratio
            test_mean_loss = 0.0
            model.eval()
            for b in range(num_test_batches):
                x = X_test[b]
                y = Y_test[b]
                out = model(x)
                test_loss = criterion(out * mask_test, y * mask_test)
                test_mean_loss += test_loss.item()
            log['test_loss'][i] = (test_mean_loss / num_test_batches)
            log['test_loss'][i] *= A_ratio
            print('test_loss =', log['test_loss'][i])

    return log 

if __name__ == '__main__':
    hidden = 128 #128
    blocks = 12 #12
    num_epochs = 10
    batch_size = 32
    log = train(hidden, blocks, num_epochs, batch_size)
    plot_log(log)





    
    
