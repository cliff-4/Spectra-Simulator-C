#!/usr/bin/env python
'''
Filename: script.py
Author: Siddharth Dhanpal
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import os.path
# import skimage.measure
import errno
import argparse
import json

try:
    import torch_mpi
except:
    torch_mpi = None

try:
    import torch_ccl
except:
    torch_ccl = None
torch.cuda.get_arch_list()
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)

# creating directory and path 
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

local_rank = 0
world_size = 1
device = None

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#if torch_ccl and int(os.environ.get('PMI_SIZE', '0')) > 1:
if int(os.environ.get('PMI_SIZE', '0')) > 1:
    print("Trying MPI dist run")
    os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')
    torch.distributed.init_process_group(backend="mpi")
    device = torch.device("cpu")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    print('local_rank', local_rank )
    print('world_size', world_size )
    print("Using CCL dist run")
else:
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Setting config for torch
#torch.set_num_threads(56)
#torch.set_num_interop_threads(2)
#torch.set_num_threads(28)

torch.set_num_interop_threads(1)

# num_train      = 1164600
# num_test       = 10000
# num_validation = 50000
# examples       = num_train + num_validation + num_test

bins_dnu = np.array(list(np.arange(1.,19.,0.1))+[19.01]) 
bins_dnu[0] = 0.99
bins_dp = np.array(list(np.arange(40.,152.5,2.5))+list(np.arange(157.,500.,7.))+[501.])
bins_dp[0] = 39.99
bins_q = np.array(list(np.arange(0.,0.74,0.02))+[0.751])
bins_q[0] = -0.01
bins_gamma = np.array(list(np.arange(0.01,0.335,0.025))+[0.351])
bins_gamma[0] = 0.005
bins_numax = np.linspace(5.,286.,100)
bins_acr = np.linspace(0.05,2.8,29)
bins_acr[0] = 0.04
bins_acr[-1] = 2.81
# bins_aer = np.linspace(0.005,0.4,18)
# bins_aer[0] = 0.004
# bins_aer[-1] = 0.41

# bins_aer = np.array(list(bins_aer[0:10]) + [bins_aer[11]] + [bins_aer[13]] + [bins_aer[15]]+ [bins_aer[17]])

bins_aer = np.linspace(0.1,0.4,12)
bins_aer[0] = 0.099
bins_aer[-1] = 0.401

bins_a3 = np.array([-0.41,-0.20,0.00,0.20,0.41])

bins_inc = np.linspace(0.,90.,19)
bins_inc[0] = -0.01
bins_inc[-1] = 90.01
bins_epp = np.linspace(0.,1.,21)
bins_epp[0] = -0.01
bins_epp[-1] = 1.01
bins_epg = np.linspace(0.,1.,11)
bins_epg[0] = -0.01 
bins_epg[-1] = 1.01
bins_snr = np.linspace(8.0, 154.13,17)
bins_snr[0] = 2.0
bins_snr[-1] = 170.
bins_vl1 = np.linspace(0.3, 2.5,12)
bins_vl1[0] = 0.29
bins_vl1[-1] = 2.51
bins_vl2 = np.linspace(0.15, 0.8,11)
bins_vl2[0] = 0.14
bins_vl2[-1] = 0.81
bins_vl3 = np.linspace(0., 0.1,11)
bins_vl3[0] = -0.01
bins_vl3[-1] = 0.11

num_classes_dnu = bins_dnu.shape[0]-1 
num_classes_dp  = bins_dp.shape[0]-1 
num_classes_q   = bins_q.shape[0]-1 
num_classes_acr = bins_acr.shape[0]-1#28 
num_classes_aer = bins_aer.shape[0]-1#11
num_classes_a3  = bins_a3.shape[0]-1#4 
num_classes_inc = bins_inc.shape[0]-1#18
num_classes_epp = bins_epp.shape[0]-1#20
num_classes_epg = bins_epg.shape[0]-1#10
num_classes_numax = bins_numax.shape[0]-1
num_classes_snr = bins_snr.shape[0]-1#16
num_classes_gamma = bins_gamma.shape[0]-1
num_classes_vl1 = bins_vl1.shape[0]-1#11
num_classes_vl2 = bins_vl2.shape[0]-1#10
num_classes_vl3 = bins_vl3.shape[0]-1#10

checkpoint_number = 0
num_epochs     = 1
num_batchsize  = 1
loss           = 'sparse_categorical_crossentropy'
pos_enc        = False 

'''Creating Path for the network model'''
parser = argparse.ArgumentParser()
parser.add_argument('--path', help="Argument for path",type=str)
parser.add_argument('--restore', help="Argument for restore",type=str)
args = parser.parse_args()
path = args.path
restore = args.restore
print('path: ',path)
print('restore: ',restore)

#path = '/export/tifr/lustre/rgdata_siddharth/models/expts/test_model_test_multiparam_torch_cnn_lstm_dense_group_norm_num_epochs_50_bs_1024'
#data_dir = '/export/tifr/lustre/rgdata_siddharth/data/data_X_files_all'
with open('training_code/config.json', 'r') as f:
    data_dir = json.load(f)['data_dir']

print(path)

mkdir_p(path)
print('path created')


class Model(nn.Module):
    '''
    Creating a CNN-LSTM-Dense model for single parameter output.
    The input has shape (None,X.25480,1) and output has classes
    of 50. This test model has only single output parameter.
    It will be extended to multiple outputs.
    Positional encoding is set to false in the 
    default case. The kernel size, strides and pool sizes are set 
    inside the function.
    '''
    def __init__(self, num_classes_dnu,num_classes_dp,num_classes_q,num_classes_acr,num_classes_aer,num_classes_a3,num_classes_inc,num_classes_epp,num_classes_epg,num_classes_numax,num_classes_snr,num_classes_gamma,num_classes_vl1,num_classes_vl2,num_classes_vl3,shape):
        super().__init__()
        kernel_size = 51#201
        strides = 1
        pool_size = 3
        filters = 16
        dilation_rate = 1
        self.shape = shape
        self.num_classes_dnu = num_classes_dnu
        self.num_classes_dp  = num_classes_dp
        self.num_classes_q   = num_classes_q
        self.num_classes_acr = num_classes_acr
        self.num_classes_aer = num_classes_aer
        self.num_classes_a3  = num_classes_a3
        self.num_classes_inc = num_classes_inc
        self.num_classes_epp = num_classes_epp
        self.num_classes_epg = num_classes_epg
        self.num_classes_numax = num_classes_numax
        self.num_classes_snr = num_classes_snr
        self.num_classes_gamma = num_classes_gamma
        self.num_classes_vl1 = num_classes_vl1
        self.num_classes_vl2 = num_classes_vl2
        self.num_classes_vl3 = num_classes_vl3

        conv_padding = int((kernel_size-1)/2)
        conv_dilation_padding = int(dilation_rate*(kernel_size-1)/2)
        pool_padding = int((pool_size-1)/2)
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels=filters, kernel_size = kernel_size, padding=conv_dilation_padding, dilation=dilation_rate, padding_mode='zeros')
        #self.bn1 = nn.BatchNorm1d(filters)
        self.gn1 = nn.GroupNorm(filters,filters)
        self.conv2 = nn.Conv1d(in_channels = filters, out_channels=2*filters, kernel_size = kernel_size, padding=conv_dilation_padding, dilation=dilation_rate, padding_mode='zeros')
        #self.bn2 = nn.BatchNorm1d(2*filters)
        self.gn2 = nn.GroupNorm(2*filters,2*filters)
        self.conv3 = nn.Conv1d(in_channels = 2*filters, out_channels=4*filters, kernel_size = kernel_size, padding=conv_padding, padding_mode='zeros')
        #self.bn3 = nn.BatchNorm1d(4*filters)
        self.gn3 = nn.GroupNorm(4*filters,4*filters)
        self.conv4 = nn.Conv1d(in_channels = 4*filters, out_channels=8*filters, kernel_size = kernel_size, padding=conv_padding, padding_mode='zeros')
        #self.bn4 = nn.BatchNorm1d(8*filters)
        self.gn4 = nn.GroupNorm(8*filters,8*filters)
        self.conv5 = nn.Conv1d(in_channels = 8*filters, out_channels=16*filters, kernel_size = kernel_size, padding=conv_padding, padding_mode='zeros')
        #self.bn5 = nn.BatchNorm1d(16*filters)
        self.gn5 = nn.GroupNorm(16*filters,16*filters)
        self.conv6 = nn.Conv1d(in_channels = 16*filters, out_channels=32*filters, kernel_size = kernel_size, padding=conv_padding, padding_mode='zeros')
        #self.bn6 = nn.BatchNorm1d(32*filters)
        self.gn6 = nn.GroupNorm(32*filters,32*filters)       
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size,stride=1,padding=1)
        self.avgpool = nn.AvgPool1d(kernel_size=pool_size,padding=1)
        self.drop = nn.Dropout(p=0.25,inplace=True)        

        self.lstm1 = nn.LSTM(batch_first=True,input_size=512,hidden_size=256)
        self.lstm2 = nn.LSTM(batch_first=True,input_size=256,hidden_size=512)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.linear_input_neurons(shape), out_features=200)
        self.linear_dnu = nn.Linear(in_features=200, out_features=num_classes_dnu)
        self.linear_dp  = nn.Linear(in_features=200, out_features=num_classes_dp)
        self.linear_q   = nn.Linear(in_features=200, out_features=num_classes_q)
        self.linear_acr = nn.Linear(in_features=200, out_features=num_classes_acr)
        self.linear_aer = nn.Linear(in_features=200, out_features=num_classes_aer)
        self.linear_a3  = nn.Linear(in_features=200, out_features=num_classes_a3)
        self.linear_inc = nn.Linear(in_features=200, out_features=num_classes_inc)
        #self.linear_epp = nn.Linear(in_features=200, out_features=num_classes_epp)
        #self.linear_epg = nn.Linear(in_features=200, out_features=num_classes_epg)
        # self.linear_numax = nn.Linear(in_features=200, out_features=num_classes_numax)
        #self.linear_snr = nn.Linear(in_features=200, out_features=num_classes_snr)
        #self.linear_gamma = nn.Linear(in_features=200, out_features=num_classes_gamma)
        #self.linear_vl1 = nn.Linear(in_features=200, out_features=num_classes_vl1)
        #self.linear_vl2 = nn.Linear(in_features=200, out_features=num_classes_vl2)
        #self.linear_vl3 = nn.Linear(in_features=200, out_features=num_classes_vl3)

       
  
    def forward(self, x):
        x = self.avgpool(self.maxpool(self.gn1(F.relu(self.conv1(torch.log(x))))))
        x = self.avgpool(self.maxpool(self.gn2(F.relu(self.conv2(x)))))
        x = self.avgpool(self.maxpool(self.gn3(F.relu(self.conv3(x)))))
        x = self.avgpool(self.maxpool(self.gn4(F.relu(self.conv4(x)))))
        x = self.avgpool(self.maxpool(self.gn5(F.relu(self.conv5(x)))))
        x = self.drop(x)
        x = self.avgpool(self.maxpool(self.gn6(F.relu(self.conv6(x)))))
        x = self.drop(x)        
        x = torch.permute(x, (0,2,1))
        x = self.lstm1(x)
        x = self.drop(x[0])        
        x = self.lstm2(x)
        x = self.drop(x[0])
        x = self.flatten(x)
        x = self.drop(x)
        x_dnu = self.linear_dnu(torch.tanh(self.linear(x)))
        x_dp  = self.linear_dp(torch.tanh(self.linear(x)))
        x_q   = self.linear_q(torch.tanh(self.linear(x)))
        x_acr = self.linear_acr(torch.tanh(self.linear(x)))
        x_aer = self.linear_aer(torch.tanh(self.linear(x)))
        x_a3 = self.linear_a3(torch.tanh(self.linear(x)))
        x_inc = self.linear_inc(torch.tanh(self.linear(x)))
        #x_epp = self.linear_epp(torch.tanh(self.linear(x)))
        #x_epg = self.linear_epg(torch.tanh(self.linear(x)))
        # x_numax= self.linear_numax(torch.tanh(self.linear(x)))
        #x_snr  = self.linear_snr(torch.tanh(self.linear(x)))
        #x_gamma= self.linear_gamma(torch.tanh(self.linear(x)))
        #x_vl1 = self.linear_vl1(torch.tanh(self.linear(x)))
        #x_vl2 = self.linear_vl2(torch.tanh(self.linear(x)))
        #x_vl3 = self.linear_vl3(torch.tanh(self.linear(x)))
        
        return x_dnu,x_dp,x_q,x_acr,x_aer,x_a3,x_inc#x_acr,x_aer,x_inc,x_epp,x_epg,x_numax,x_snr,x_gamma,x_vl1,x_vl2,x_vl3        
        
    def size_after_cnnlstm(self, x):
        x = self.avgpool(self.maxpool(self.gn1(F.relu(self.conv1(x)))))
        x = self.avgpool(self.maxpool(self.gn2(F.relu(self.conv2(x)))))
        x = self.avgpool(self.maxpool(self.gn3(F.relu(self.conv3(x)))))
        x = self.avgpool(self.maxpool(self.gn4(F.relu(self.conv4(x)))))
        x = self.avgpool(self.maxpool(self.gn5(F.relu(self.conv5(x)))))
        x = self.avgpool(self.maxpool(self.gn6(F.relu(self.conv6(x)))))
        x = torch.permute(x, (0,2,1))
        x = self.lstm1(x)
        x = self.lstm2(x[0])
        x = self.flatten(x[0])
        
        
        return x.size()
    
    def linear_input_neurons(self,shape):
        size = self.size_after_cnnlstm(torch.rand(1, 1, shape)) 
        m = 1
        for i in size:
            m *= i

        return int(m)
    
    
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print(num_classes_dnu,num_classes_dp,num_classes_q,num_classes_acr,num_classes_aer,num_classes_a3,num_classes_inc,num_classes_epp,num_classes_epg,num_classes_numax,num_classes_snr,num_classes_gamma,num_classes_vl1,num_classes_vl2,num_classes_vl3)
shape = 35692
model = Model(num_classes_dnu,num_classes_dp,num_classes_q,num_classes_acr,num_classes_aer,num_classes_a3,num_classes_inc,num_classes_epp,num_classes_epg,num_classes_numax,num_classes_snr,num_classes_gamma,num_classes_vl1,num_classes_vl2,num_classes_vl3,shape).to(device)
best_val_loss = 1.0e8
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.0005, steps_per_epoch=20, epochs=1)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.3)

model = model.float()

#Uncomment following lines of script if you want to load a checkpoint
if restore=='yes':
    model_dir = path
    checkpoint = torch.load(model_dir+'/is_best.pth')
    checkpoint_number = checkpoint['epoch']+1
    num_epochs = num_epochs - checkpoint_number
    for key in list(checkpoint['model_state_dict'].keys()):
        new_key = key[7:]
        checkpoint['model_state_dict'][new_key] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_loss = checkpoint['val_loss'].item()


if world_size > 1:
    model = DDP(model)


def scale_parameter(a):
    '''Scales a column into 0-1'''
    a = (a-a.min())/(a.max()-a.min())
    return a

def categorize(par,num_cl):
    '''Categorizes a scaled parameter into given number of classes'''
    par = par*num_cl
    par = np.floor(par)
    par = par.astype('int')
    par = np.where(par<num_cl,par,num_cl-1)
    #par = to_categorical(par,num_classes=num_cl)
    return par

print_freq = 30
start_time_training=time.time()
X_handle = None
for epoch in range(checkpoint_number,checkpoint_number+num_epochs):  # loop over the dataset multiple times
    avg_training_loss_epoch = 0
    avg_val_loss_epoch = 0
    temp_train_idx = 0
    temp_val_idx = 0

    for dataset_number in list(range(local_rank, 144, world_size)):
#####################################################################################################################    
            
        start_time=time.time()
        print(start_time)
        print(f"loading data: {data_dir + '/data_'+'%03d'%dataset_number+'.npy'}")
        data = np.load(data_dir + '/data_'+'%03d'%dataset_number+'.npy').astype('float32')
        # expected_shape = (-1, 35692, 36)
        # data = data.reshape(expected_shape)
        print(f"Data shape after memmap: {data.shape}")
        if X_handle is not None: X_handle.close()
        X_handle = open(data_dir + '/data_'+'%03d'%dataset_number+'.npy', 'rb')
        end_time=time.time()
        # print(X.shape)
        print(end_time)
        print('Time taken for loading data= %f s'%(end_time-start_time))

        Y = []
        X = []

        for i in range(0,5120):
#            if (data[i,-1]>=0.45):
            Y.append(data[i,-36:])
            X.append(data[i,:35692])
        
        print(f"Data shape: {data.shape}")
        print(f"Min value in last column: {data[:, -1].min()}, Max value: {data[:, -1].max()}")

        for i in range(0, 5120):
            if (data[i, -1] >= 0.45):  # Adjust threshold if needed
                Y.append(data[i, -36:])
                X.append(data[i, :35692])

        if len(X) == 0:
            raise ValueError("No rows satisfy the condition. Check the data or adjust the filtering criteria.")

        # Y = data[:5120,-36:]
        # X = data[:5120,:35692]

        Y = np.array(Y[:4336])
        X = np.array(X[:4336])
        
        # print(X.shape,Y.shape)

        print(f"Shape of X before reshape: {X.shape}")        
        if len(X.shape) < 2:
            if X.size == 0:
                raise ValueError("X is empty. Check the data loading process.")
            raise ValueError(f"X has an unexpected shape: {X.shape}. Ensure the data is loaded correctly.")
        X = X.reshape((X.shape[0],X.shape[1],1))
        print(X.shape,Y.shape)

        print('dataset preparation completed')

        aer, acr, a3, Dnu, Dp, q, inc= Y[:,0], Y[:,1], Y[:,6]/Y[:,0], Y[:,12], Y[:,17], Y[:,19], Y[:,-1]
        # aer, acr, a3, Dnu, epsilon_p, Dp, epsilon_g ,q= Y[:,0], Y[:,1], Y[:,6], Y[:,12], Y[:,13], Y[:,17], Y[:,18], Y[:,19]
        # snr = Y[:,20]
        # gamma = Y[:,21]
        # vl1,vl2,vl3 = Y[:,23],Y[:,24],Y[:,25]
        # numax,inc = Y[:,-2],Y[:,-1]
        
        Dnu = (np.digitize(Dnu,bins_dnu)-1).astype('int')
        Dp  = (np.digitize(Dp,bins_dp)-1).astype('int')
        q   = (np.digitize(q,bins_q)-1).astype('int')
        # numax = (np.digitize(numax,bins_numax)-1).astype('int')
        # gamma = (np.digitize(gamma,bins_gamma)-1).astype('int')
        acr = (np.digitize(acr,bins_acr)-1).astype('int')
        aer = (np.digitize(aer,bins_aer)-1).astype('int')
        a3 = (np.digitize(a3,bins_a3)-1).astype('int')
        inc = (np.digitize(inc,bins_inc)-1).astype('int')
        # snr = (np.digitize(snr,bins_snr)-1).astype('int')
        # vl1 = (np.digitize(vl1,bins_vl1)-1).astype('int')
        # vl2 = (np.digitize(vl2,bins_vl2)-1).astype('int')
        # vl3 = (np.digitize(vl3,bins_vl3)-1).astype('int')
        # epsilon_p = (np.digitize(epsilon_p,bins_epp)-1).astype('int')
        # epsilon_g  = (np.digitize(epsilon_g,bins_epg)-1).astype('int')

        Dnu = np.clip(Dnu, 0, num_classes_dnu-1)
        Dp = np.clip(Dp, 0, num_classes_dp-1)
        q = np.clip(q, 0, num_classes_q-1)
        acr = np.clip(acr, 0, num_classes_acr-1)
        aer = np.clip(aer, 0, num_classes_aer-1)
        a3 = np.clip(a3, 0, num_classes_a3-1)
        inc = np.clip(inc, 0, num_classes_inc-1)


        #string = ['Dp']
        #labels_train = Dp_train#[Dnu_train,Dp_train]
        #labels_val   = Dp_val#[Dnu_val,Dp_val]
        string = ['Dnu','Dp','q','acr','aer','a3','inc']#['Dnu','Dp','q','acr','aer','inc','epp','epg','numax','snr','gamma','vl1','vl2','vl3']
        labels_train = [Dnu,Dp,q,acr,aer,a3,inc]#[Dnu,Dp,q,acr,aer,inc,epsilon_p,epsilon_g,numax,snr,gamma,vl1,vl2,vl3]#Dp_train#[Dnu_train,Dp_train]


        X = np.transpose(X, (0, 2, 1))
        labels_train = np.array(labels_train)
        labels_train = labels_train.T#reshape((labels_train.shape[1],labels_train.shape[0]))
        print(labels_train.shape)
        X_train_labels = np.empty((X.shape[0],X.shape[1],X.shape[2]+len(string)),dtype=np.float32)
        X_train_labels[:,0,-len(string):] = labels_train
        X_train_labels[:,:,:-len(string)] = X
        print(X_train_labels.shape)

        batch_size = num_batchsize // world_size

        trainloader = torch.utils.data.DataLoader(X_train_labels, batch_size=batch_size,shuffle=False)#, num_workers=2)
    
##############################################################################################################    
    
        correct_pred = np.zeros((len(string),)) 
        total_pred = np.zeros((len(string),))
        running_loss = 0.0
        running_loss1 = 0.0
        start_time=time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[:,:,:-len(string)],data[:,:,-len(string):]
            inputs = inputs.type(torch.FloatTensor)


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())

            labels_temp = labels[:,:,0].type(torch.long)

            labels_temp = torch.squeeze(labels_temp, dim = 1)      

            loss = criterion(outputs[0], labels_temp)

            for k in range(1,len(string)):
                #print(k,string[k])
                labels_temp = labels[:,:,k].type(torch.long)
                labels_temp = torch.squeeze(labels_temp, dim = 1)        
                loss+=criterion(outputs[k], labels_temp)
            loss.backward()
            avg_training_loss_epoch+=loss.item()
            temp_train_idx+=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss1 += loss.item()
            if (i+1) % print_freq == 0:    # print every 3 mini-batches
                end_time=time.time()
                print('[%d, %3d, %5d] loss: %.6f  lbs: %d  gbs: %d  time/iter: %.3f s  Training time so far: %8.1f s' %
                      (epoch + 1, dataset_number, i + 1, running_loss / print_freq, batch_size, batch_size * world_size, (end_time-start_time)/print_freq, end_time-start_time_training))
                #print('Avg. time/iter taken for %d iterations= %f s'%(print_freq, (end_time-start_time)/print_freq))
                start_time=time.time()
                running_loss = 0.0

        print(temp_train_idx, avg_training_loss_epoch, i+1, running_loss1/(i+1) )
        del X
        del Y
        del X_train_labels
    scheduler.step()

    print('Average loss in the epoch = %.6f'%(avg_training_loss_epoch/temp_train_idx))
    avg_training_loss_epoch_datasets = torch.tensor(avg_training_loss_epoch/temp_train_idx)
    torch.distributed.all_reduce(avg_training_loss_epoch_datasets,op=torch.distributed.ReduceOp.SUM)
    print('Average loss in the epoch across datasets = %.6f'%(avg_training_loss_epoch_datasets.item()/world_size))
    training_loss = avg_training_loss_epoch_datasets/world_size

    for dataset_number in list(range(144+local_rank,152,world_size)):
#####################################################################################################################    
        start_time=time.time()
        print(start_time)
        data = np.load(data_dir + '/data_'+'%03d'%dataset_number+'.npy')
        if X_handle is not None: X_handle.close()
        X_handle = open(data_dir + '/data_'+'%03d'%dataset_number+'.npy', 'rb')
        end_time=time.time()
        # print(X.shape)
        print(end_time)
        print('Time taken for loading data= %f s'%(end_time-start_time))

        Y = []
        X = []

        for i in range(0,5120):
            if (data[i,-1]>=0.45):
                Y.append(data[i,-36:])
                X.append(data[i,:35692])

        # Y = data[:5120,-36:]
        # X = data[:5120,:35692]

        Y = np.array(Y[:4336])
        X = np.array(X[:4336])
        
        # print(X.shape,Y.shape)
        
        X = X.reshape((X.shape[0],X.shape[1],1))
        print(X.shape,Y.shape)

        print('dataset preparation completed')

        aer, acr, a3, Dnu, Dp, q, inc= Y[:,0], Y[:,1], Y[:,6]/Y[:,0], Y[:,12], Y[:,17], Y[:,19], Y[:,-1] 

        # snr = Y[:,10]

        # gamma = Y[:,11]

        # vl1,vl2,vl3 = Y[:,13],Y[:,14],Y[:,15]
        # numax,inc = Y[:,-2],Y[:,-1]

        
        Dnu = (np.digitize(Dnu,bins_dnu)-1).astype('int')
        Dp  = (np.digitize(Dp,bins_dp)-1).astype('int')
        q   = (np.digitize(q,bins_q)-1).astype('int')
        # numax = (np.digitize(numax,bins_numax)-1).astype('int')
        # gamma = (np.digitize(gamma,bins_gamma)-1).astype('int')
        acr = (np.digitize(acr,bins_acr)-1).astype('int')
        aer = (np.digitize(aer,bins_aer)-1).astype('int')
        a3 = (np.digitize(a3,bins_a3)-1).astype('int')
        inc = (np.digitize(inc,bins_inc)-1).astype('int')
        # snr = (np.digitize(snr,bins_snr)-1).astype('int')
        # vl1 = (np.digitize(vl1,bins_vl1)-1).astype('int')
        # vl2 = (np.digitize(vl2,bins_vl2)-1).astype('int')
        # vl3 = (np.digitize(vl3,bins_vl3)-1).astype('int')
        # epsilon_p = (np.digitize(epsilon_p,bins_epp)-1).astype('int')
        # epsilon_g  = (np.digitize(epsilon_g,bins_epg)-1).astype('int')

        # Add this after the digitization to ensure no negative indices
        Dnu = np.clip(Dnu, 0, num_classes_dnu-1)
        print(f"Dnu: {Dnu.min()}, {Dnu.max()}")
        Dp = np.clip(Dp, 0, num_classes_dp-1)
        print(f"Dp: {Dp.min()}, {Dp.max()}")
        q = np.clip(q, 0, num_classes_q-1)
        print(f"q: {q.min()}, {q.max()}")
        acr = np.clip(acr, 0, num_classes_acr-1)
        print(f"acr: {acr.min()}, {acr.max()}")
        aer = np.clip(aer, 0, num_classes_aer-1)
        print(f"aer: {aer.min()}, {aer.max()}")
        a3 = np.clip(a3, 0, num_classes_a3-1)
        print(f"a3: {a3.min()}, {a3.max()}")
        inc = np.clip(inc, 0, num_classes_inc-1)
        print(f"inc: {inc.min()}, {inc.max()}")


        string = ['Dnu','Dp','q','acr','aer','a3','inc']#['Dnu','Dp','q','acr','aer','inc','epp','epg','numax','snr','gamma','vl1','vl2','vl3']
        labels_val = [Dnu,Dp,q,acr,aer,a3,inc]#[Dnu,Dp,q,acr,aer,inc,epsilon_p,epsilon_g,numax,snr,gamma,vl1,vl2,vl3]#Dp_train#[Dnu_train,Dp_train]


        X = np.transpose(X, (0, 2, 1))
        labels_val = np.array(labels_val)
        labels_val = labels_val.T#reshape((labels_train.shape[1],labels_train.shape[0]))
        print(labels_val.shape)
        X_val_labels = np.empty((X.shape[0],X.shape[1],X.shape[2]+len(string)),dtype=np.float32)
        X_val_labels[:,0,-len(string):] = labels_val
        X_val_labels[:,:,:-len(string)] = X
        print(X_val_labels.shape)

        batch_size = num_batchsize // world_size

        valloader = torch.utils.data.DataLoader(X_val_labels, batch_size=batch_size,shuffle=False)#, num_workers=2)
    
##############################################################################################################    
    
        correct_pred = np.zeros((len(string),)) 
        total_pred = np.zeros((len(string),))
        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data[:,:,:-len(string)],data[:,:,-len(string):]
            inputs = inputs.type(torch.FloatTensor)

            # forward + backward + optimize
            outputs = model(inputs.float())

            labels_temp = labels[:,:,0].type(torch.long)

            labels_temp = torch.squeeze(labels_temp, dim = 1)

            loss = criterion(outputs[0], labels_temp)

            for k in range(1,len(string)):
                #print(string[k])
                labels_temp = labels[:,:,k].type(torch.long)
                labels_temp = torch.squeeze(labels_temp, dim = 1)
                loss+=criterion(outputs[k], labels_temp)

            avg_val_loss_epoch+=loss.item()
            temp_val_idx+=1

            # print statistics
            running_loss += loss.item()
            if (i+1) % print_freq == 0:    # print every 3 mini-batches
                end_time=time.time()
                print('[%d, %3d, %5d] val loss: %.6f  lbs: %d  gbs: %d  time/iter: %.3f s  Training time so far: %8.1f s' %
                      (epoch + 1, dataset_number, i + 1, running_loss / print_freq, batch_size, batch_size * world_size, (end_time-start_time)/print_freq, end_time-start_time_training))
                start_time=time.time()
                running_loss = 0.0

            for k in range(0,len(string)):
                _, predictions = torch.max(outputs[k], 1)
                # collect the correct predictions for each class
                labels_temp = labels[:,:,k].type(torch.long)
                labels_temp = torch.squeeze(labels_temp, dim = 1)
                for label, prediction in zip(labels_temp, predictions):
                    if label == prediction:
                        correct_pred[k]+=1
                    total_pred[k]+=1

        for k in range(0,len(string)):
            print('Dataset number: %d Val accuracy '%dataset_number + string[k]+': %.3f'%(correct_pred/total_pred)[k])

        
        del X
        del Y
        del X_val_labels

    print('Average validation loss in the epoch = %.6f'%(avg_val_loss_epoch/temp_val_idx))
    avg_val_loss_epoch_datasets = torch.tensor(avg_val_loss_epoch/temp_val_idx)
    torch.distributed.all_reduce(avg_val_loss_epoch_datasets,op=torch.distributed.ReduceOp.SUM)
    print('Average validation loss in the epoch across datasets = %.6f'%(avg_val_loss_epoch_datasets.item()/world_size))
    val_loss = avg_val_loss_epoch_datasets/world_size
    
    if local_rank == 0:
        torch.save({'epoch': epoch,'optimizer_state_dict': optimizer.state_dict(),'train_loss': training_loss,'val_loss': val_loss,'model_state_dict': model.state_dict(),'scheduler_state_dict':scheduler.state_dict()},path+'/checkpoint.pth')
        torch.save({'epoch': epoch,'optimizer_state_dict': optimizer.state_dict(),'train_loss': training_loss,'val_loss': val_loss,'model_state_dict': model.state_dict(),'scheduler_state_dict':scheduler.state_dict()},path+'/checkpoint-%d.pth'%epoch)
        if val_loss.item()< best_val_loss:
            best_val_loss =val_loss.item()
            torch.save({'epoch': epoch,'optimizer_state_dict': optimizer.state_dict(),'train_loss': training_loss,'val_loss': val_loss,'model_state_dict': model.state_dict(),'scheduler_state_dict':scheduler.state_dict()},path+'/is_best.pth')

    


end_time_training=time.time()
print('Time taken for total training= %f s'%(end_time_training-start_time_training))


print('Finished Training')
if local_rank == 0:
    torch.save({'epoch': epoch,'optimizer_state_dict': optimizer.state_dict(),'train_loss': training_loss,'val_loss': val_loss,'model_state_dict': model.state_dict(),'scheduler_state_dict':scheduler.state_dict()},path+'/model.pth') 

print('Finished saving')
