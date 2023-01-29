import os 
import pickle 
import numpy as np 
import torch
from tqdm import tqdm 
                  
def create_train_local(IND = 0, nn = 5):
     X = [] # num_A * files X (nn X 3)
     Y = [] # num_A * files 
     dir = 'train_data'
     
     directory = os.fsencode(dir)
     perm = [[0,1,2], [0,2,1], [1,0,2], [2,0,1], [2,1,0], [1,2,0]]
     ref = np.array([[1,1,1], [-1,1,1], [1,-1,1], [1,-1,1]])
     dir_list = os.listdir(directory)
     num_files = len(dir_list)
     for i in tqdm(range(num_files)):
          filename = os.fsdecode(dir_list[i])
          with open(dir + '/' + filename, 'rb') as handle:
               f = pickle.load(handle)
          typ = f['types']
          y = f['targets'][IND]
          x = f['positions']
          N = len(typ) 
          for i in range(N):
               if typ[i] == 0:
                    x_curr = x[i,:]
                    y_curr = y[i]
                    R = x - x_curr[None,:]
                    dis = np.sqrt(np.sum(R * R, axis = 1))
                    ind = np.argsort(-dis)[:nn]
                    R_n = R[ind,:]
                    for p in perm:
                         for r in ref:
                              Y.append(y_curr) 
                              X.append((R_n[:,p] * r[None,:]).flatten())
     return X, Y, dir 


def create_test_local(IND = 0, nn = 5):
     X = [] # num_A * files X (nn X 3)
     Y = [] # num_A * files 
     dir = 'test_data'
     directory = os.fsencode(dir)
     for file in os.listdir(directory):
          filename = os.fsdecode(file)
          with open(dir + '/' + filename, 'rb') as handle:
               f = pickle.load(handle)
          typ = f['types']
          x = f['positions']
          y = f['targets'][IND]
          N = len(typ)
          
          for i in range(N):
               if typ[i] == 0:
                    x_curr = x[i,:]
                    y_curr = y[i]
                    r = x - x_curr[None,:]
                    dis = np.sqrt(np.sum(r * r, axis = 1))
                    ind = np.argsort(-dis)[:nn]
                    x_n = r[ind,:].flatten()
                    
                    X.append(x_n)
                    Y.append(y_curr)
     return X, Y, dir 
          

def create_train_test(IND, mode, nn=5):
     if mode == 'test':
          X, Y, dir = create_test_local(IND)
     else:
          X, Y, dir = create_train_local(IND)

     X = np.array(X)
     Y = np.array(Y)
     ind = np.arange(X.shape[0])
     np.random.shuffle(ind)
     X = X[ind]
     Y = Y[ind]
     X = torch.tensor(X).float()
     Y = torch.tensor(Y).float()
     print(X.shape)
     data_dict = {'X':X, 'Y':Y}
     with open(dir + '_dict_local.pickle', 'wb') as handle:
          pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


IND = 0
nn = 50
create_train_test(IND, 'test', nn)
create_train_test(IND, 'train', nn)
          
     