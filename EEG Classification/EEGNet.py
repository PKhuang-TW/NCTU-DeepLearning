import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.optim as optim
from tqdm import tqdm
import pandas as pd


# # Load Data

class trainDataset(Data.Dataset):
    def __init__(self):
        S4b_train = np.load('S4b_train.npz')
        X11b_train = np.load('X11b_train.npz')

        self.train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
        self.train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)

        self.train_label = self.train_label - 1
        self.train_data = np.transpose(np.expand_dims(self.train_data, axis=1), (0, 1, 3, 2))

        mask = np.where(np.isnan(self.train_data))
        self.train_data[mask] = np.nanmean(self.train_data)
        
        self.len = len(self.train_data)
    
    def __getitem__(self, index):
        return self.train_data[index], self.train_label[index]
    
    def __len__(self):
        return self.len
    
class testDataset(Data.Dataset):
    def __init__(self):
        S4b_test = np.load('S4b_test.npz')
        X11b_test = np.load('X11b_test.npz')

        self.test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
        self.test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

        self.test_label = self.test_label -1
        self.test_data = np.transpose(np.expand_dims(self.test_data, axis=1), (0, 1, 3, 2))

        mask = np.where(np.isnan(self.test_data))
        self.test_data[mask] = np.nanmean(self.test_data)
        
        self.len = len(self.test_data)
    
    def __getitem__(self, index):
        return self.test_data[index], self.test_label[index]
    
    def __len__(self):
        return self.len


device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 64
EPOCH = 600
lr = 0.01
TensorInputType = torch.FloatTensor
TensorLabelType = torch.LongTensor

trainData = trainDataset()
train_loader = Data.DataLoader(
    dataset = trainData,
    batch_size = BATCH,
    shuffle = True,
    )

testData = testDataset()
test_loader = Data.DataLoader(
    dataset = testData,
    )


# # Neural Network

activations = {
    'ReLU': nn.ReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'ELU': nn.ELU()
}

class EEGNet(nn.Module):
    def __init__(self, activation_function, dropout):
        super(EEGNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=(1,51),      # filter size
                stride=1,           # filter movement/step
                padding=(0, 25),
            ), 
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, (2,1), 1, groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation_function],
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p=dropout)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1,15), 1, padding=(0,7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activations[activation_function],
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8), padding=0),
            nn.Dropout(p=dropout)
        )
        self.out = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        x = self.conv1(x)     
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


# # Evaluate Data

def EEGtrain(activation_function, opt, dropout):
    train_acc = []
    valid_acc = []
    
    eegnet = EEGNet(activation_function, dropout).to(device)
    #print(eegnet)  # net architecture
    criterion = nn.CrossEntropyLoss()
    
    if(opt == 'Adam'):
        optimizer = torch.optim.Adam(eegnet.parameters(), lr=lr)
    elif(opt == 'SGD'):
        optimizer = torch.optim.SGD(eegnet.parameters(), lr=lr)
    elif(opt == 'RMSprop'):
        optimizer = torch.optim.RMSprop(eegnet.parameters(), lr=lr)
    
    for epoch in tqdm(range(EPOCH)):
        # Train
        eegnet.train()
        correct = 0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.type(TensorInputType).to(device), labels.type(TensorLabelType).to(device)
            y_pred = eegnet(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == labels).sum().item()   
        acc = 100*correct / len(trainData)
        train_acc.append(acc)
        
        # Test
        eegnet.eval()
        with torch.no_grad():
            correct = 0
            for idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.type(TensorInputType).to(device), labels.type(TensorLabelType).to(device)
                y_pred = eegnet(inputs)
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == labels).sum().item()
            acc = 100*correct / len(testData)
            valid_acc.append(acc)
            #print('loss: {:.4f}'.format(loss.item()))
            print ('Epoch [{}/{}], Train_Acc: {:.3f}, Valid_Acc: {:3f} %'.format(epoch+1, EPOCH, train_acc[-1], valid_acc[-1]))
    return train_acc, valid_acc


# # Grid Search

dropout=[0.2, 0.3, 0.4, 0.5]
activation_function = ['ReLU', 'ELU', 'LeakyReLU']
opts = ['Adam', 'SGD', 'RMSprop']
loss_function = 'CE'
best_list = []

for drop in dropout:
    for act_func in activation_function:
        for optimizer in opts:
            print('===============\n\n')
            #csvName = 'test_csv\\EEGNet_' + act_func + '_' + str(drop) + '_' + optimizer + '_' + loss_function + '_' +'.csv'
            csvName = 'csv\\EEGNet_' + act_func + '_' + str(drop) + '_' + optimizer + '_' + loss_function + '_' +'.csv'
            print('Training...')
            print(csvName.split('\\')[1].split('csv')[0][:-2])
            train_acc, valid_acc = EEGtrain(activation_function=act_func, opt=optimizer, dropout=drop)
            valid_acc = np.array(valid_acc)
            csvData = {'Train':train_acc, "Test":valid_acc}
            csvData = pd.DataFrame(csvData)
            csvData.to_csv(csvName)

