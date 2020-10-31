import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models
from torchvision import transforms
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# # Pretrain Model
class pretrain_ResNet(nn.Module):
    def __init__(self):
        super(pretrain_ResNet, self).__init__()

        self.classify = nn.Linear(2048, 5)

        pretrained_model = torchvision.models.__dict__['resnet{}'.format(50)](pretrained=True)
        self.conv1 = pretrained_model._modules['conv1']
        self.bn1 = pretrained_model._modules['bn1']
        self.relu = pretrained_model._modules['relu']
        self.maxpool = pretrained_model._modules['maxpool']

        self.layer1 = pretrained_model._modules['layer1']
        self.layer2 = pretrained_model._modules['layer2']
        self.layer3 = pretrained_model._modules['layer3']
        self.layer4 = pretrained_model._modules['layer4']

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        del pretrained_model

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)

        return x


# # w/o Pretrain Model
# convolution layer with kernel size = 3
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# convolution layer with kernel size = 1
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # filter num difference within a block

    def __init__(self, in_filterNum, o_filterNum, stride=1, downsample=None):
        '''
        Args :
            in_filterNum : Input filter Num.
            o_filterNum : Output filter Num.
        '''
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_filterNum, o_filterNum, stride)
        self.bn1 = nn.BatchNorm2d(o_filterNum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(o_filterNum, o_filterNum)
        self.bn2 = nn.BatchNorm2d(o_filterNum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # Identity Mapping

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # filter num difference within a block

    def __init__(self, in_filterNum, o_filterNum, stride=1, downsample=None):
        '''
        Args :
            in_filterNum : Input filter Num.
            o_filterNum : Output filter Num.
        '''
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_filterNum, o_filterNum)
        self.bn1 = nn.BatchNorm2d(o_filterNum)
        self.conv2 = conv3x3(o_filterNum, o_filterNum, stride)
        self.bn2 = nn.BatchNorm2d(o_filterNum)
        self.conv3 = conv1x1(o_filterNum, o_filterNum * self.expansion)
        self.bn3 = nn.BatchNorm2d(o_filterNum * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # Identity Mapping

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blockNum, num_classes=5, width_per_group=64):
        '''
        Args :
            block : BasicBlock for ResNet-18/34, Bottleneck for ResNet-50/101/152.
            blockNum : 4-int list of the number of building blocks in conv1_~conv4_
            num_classes : Number of classes
            width_per_group : The number of filter difference between each conv_
        '''
        super(ResNet, self).__init__()
        filterNums = [int(width_per_group * 2 ** i) for i in range(4)]  # [64,128,256,512]
        self.in_filterNum = filterNums[0]  # Initial filter num = 64
        
        self.conv1 = nn.Conv2d(3, filterNums[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(filterNums[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # construct conv1_~conv4_
        self.layer1 = self._make_layer(block, filterNums[0], blockNum[0])
        self.layer2 = self._make_layer(block, filterNums[1], blockNum[1], stride=2)
        self.layer3 = self._make_layer(block, filterNums[2], blockNum[2], stride=2)
        self.layer4 = self._make_layer(block, filterNums[3], blockNum[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(filterNums[3] * block.expansion, num_classes)

    def _make_layer(self, block, filterNum, blockNum, stride=1):
        '''
        Args :
            block : BasicBlock for ResNet-18/34, Bottleneck for ResNet-50/101/152.
            filterNum : The filter num in current conv_ layer.
            blockNum : The num of building block in current conv_layer.
        '''
        downsample = None
        
        # When there are different dimension, then downsample
        if stride != 1 or self.in_filterNum != filterNum * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_filterNum, filterNum * block.expansion, stride),
                nn.BatchNorm2d(filterNum * block.expansion),
            )

        layers = []
        layers.append(block(self.in_filterNum, filterNum, stride, downsample))
        self.in_filterNum = filterNum * block.expansion  # Update filter num by different building block
        for _ in range(1, blockNum):
            layers.append(block(self.in_filterNum, filterNum))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# # Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
#             title = 'Normalized confusion matrix'
            title = title
        else:
#             title = 'Confusion matrix, without normalization'
            title = title

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# # DataLoader
imgSize = 512
rotateAngle = 30

train_preprocess = transforms.Compose([
    transforms.Scale(imgSize),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(rotateAngle),
    transforms.ToTensor(),
    transforms.Normalize((0.3749, 0.2601, 0.1856), (0.2526, 0.1780, 0.1291)),
])

test_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3749, 0.2601, 0.1856), (0.2526, 0.1780, 0.1291)),
])

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('data\\csv\\train_img.csv')
        label = pd.read_csv('data\\csv\\train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    elif mode == 'test':
        img = pd.read_csv('data\\csv\\test_img.csv')
        label = pd.read_csv('data\\csv\\test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class dataset(Data.Dataset):
    def __init__(self, root=None, mode=None):
        self.mode = mode
        self.img_names, self.labels = getData(mode)
        print('> Found %d datas...' % (len(self.img_names)))

    def __getitem__(self, index):
        img_path = 'data\\imgs\\' + self.img_names[index] + '.jpeg'
        img = Image.open(img_path)
        if(self.mode=='train'):
            img = train_preprocess(img)
        elif(self.mode=='test'):
            img = test_preprocess(img)
#         img = np.transpose(img, (2,0,1))
        return img, self.labels[index]

    def __len__(self):
        ''' Return the size of dataset '''
        return len(self.labels)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
TensorInputType = torch.FloatTensor
TensorLabelType = torch.LongTensor

BATCH = 16
EPOCH = 20
lr = 0.001
momentum = 0.9
weight_decay = 5.0e-4

trainData = dataset(mode='train')
train_loader = Data.DataLoader(
    dataset = trainData,
    batch_size = BATCH,
    shuffle = True,
    )

testData = dataset(mode='test')
test_loader = Data.DataLoader(
    dataset = testData,
    batch_size = BATCH,
    )


# # Training Code
def train(net=None, momentum=momentum, weight_decay=weight_decay, pretrain=False, saveModel=False):
    
    if(net == None):
        print('There is no NN loaded')
        return None, None
    
    train_acc = []
    train_loss = []
    test_acc = []
    pred = []

    #print(net)
    optimizer = torch.optim.SGD(net.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()


    for epoch in tqdm(range(EPOCH)):
        # Train
        net.train()
        correct = 0
        print('-- [Epoch {}/{}] --'.format(epoch+1, EPOCH))
        print('training...')
        for idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.type(TensorInputType).to(device), labels.type(TensorLabelType).to(device)
            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == labels).sum().item()
        acc = 100*correct / len(trainData)
        train_acc.append(acc)
        train_loss.append(loss.item())

        # Test
        net.eval()
        with torch.no_grad():
            correct = 0
            print('testing...')
            for idx, (inputs, labels) in enumerate(tqdm(test_loader)):
                inputs, labels = inputs.type(TensorInputType).to(device), labels.type(TensorLabelType).to(device)
                y_pred = net(inputs)
                _, predicted = torch.max(y_pred.data, 1)
                if(epoch == EPOCH-1):
                    pred.append(predicted.cpu().tolist())
                correct += (predicted == labels).sum().item()
            acc = 100*correct / len(testData)
            test_acc.append(acc)
            print ('Train_Acc: {:.3f}, loss: {:.3f}, Test_Acc: {:3f} %'.format(train_acc[-1], loss.item(), test_acc[-1]))
        print('\n================\n\n')
    if(saveModel):
        modelName = time.strftime("%Y%m%d_%H%M", time.localtime())
        modelName +=  '_s' + str(imgSize) + '_r' + str(rotateAngle) + '_ep' + str(EPOCH) + '_b' + str(BATCH)
        if(pretrain):
            modelName = 'model\\ResNet50\\pretrain\\' + modelName + '.pth'
        else:
            modelName = 'model\\ResNet50\\no_pretrain\\' + modelName + '.pth'
        torch.save(net.state_dict(), modelName)
    return train_acc, train_loss, test_acc, pred


# # Main

# ## w/o Pretrain
# [20655.  1955.  4210.   698.   581.]
# [0.73507954 0.06957543 0.1498274  0.02484074 0.02067689]
pretrain = False
lr = 0.001
momentum = 0.9
weight_decay = 5.0e-4

print('Params : size={}, rotate={}, batch={}, lr={}, momentum={}, weight_decay={}'.format(imgSize, rotateAngle, BATCH, lr, momentum, weight_decay))
net = ResNet(Bottleneck, [3, 4, 6, 3]).to(device)
#net = nn.DataParallel(net, device_ids=[0,1])
train_acc, train_loss, test_acc, pred = train(net,momentum=momentum, weight_decay=weight_decay, pretrain=pretrain, saveModel=True)

curtime = time.strftime("%Y%m%d_%H%M", time.localtime())
csvName = 'csv\\ResNet50\\no_pretrain\\' + curtime + '_s' + str(imgSize) + '_r' + str(rotateAngle) + '_ep' + str(EPOCH) + '_b' + str(BATCH) + '.csv'
csvData = {'Train':train_acc, 'Train_loss':train_loss, 'Test':test_acc}
csvData = pd.DataFrame(csvData)
csvData.to_csv(csvName)

pred = sum(pred, [])
plot_confusion_matrix(getData('test')[1], pred, normalize=True, classes=np.array(['0','1','2','3','4']), title='Normalizated Confusion matrix')
cmName = 'confusionMatrix\\ResNet50\\no_pretrain\\' + curtime + '_s' + str(imgSize) + '_r' + str(rotateAngle) + '_ep' + str(EPOCH) + '_b' + str(BATCH) + '.png'
plt.savefig(cmName)
plt.show()


# ## Pretrain
# [20655.  1955.  4210.   698.   581.]
# [0.73507954 0.06957543 0.1498274  0.02484074 0.02067689]
pretrain = True
lr = 0.001
momentum = 0.9
weight_decay = 5.0e-4

print('Params : size={}, rotate={}, batch={}, lr={}, momentum={}, weight_decay={}'.format(imgSize, rotateAngle, BATCH, lr, momentum, weight_decay))
net = pretrain_ResNet().to(device)
#net = nn.DataParallel(net, device_ids=[0,1])
train_acc, train_loss, test_acc, pred = train(net,momentum=momentum, weight_decay=weight_decay, pretrain=pretrain, saveModel=True)

curtime = time.strftime("%Y%m%d_%H%M", time.localtime())
csvName = 'csv\\ResNet50\\pretrain\\' + curtime + '_s' + str(imgSize) + '_r' + str(rotateAngle) + '_ep' + str(EPOCH) + '_b' + str(BATCH) + '.csv'
csvData = {'Train':train_acc, 'Train_loss':train_loss, 'Test':test_acc}
csvData = pd.DataFrame(csvData)
csvData.to_csv(csvName)

pred = sum(pred, [])
plot_confusion_matrix(getData('test')[1], pred, normalize=True, classes=np.array(['0','1','2','3','4']), title='Normalizated Confusion matrix')
cmName = 'confusionMatrix\\ResNet50\\pretrain\\' + curtime + '_s' + str(imgSize) + '_r' + str(rotateAngle) + '_ep' + str(EPOCH) + '_b' + str(BATCH) + '.png'
plt.savefig(cmName)
plt.show()