
from pandas.core.frame import DataFrame
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.models import resnet50

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

import os
from torchvision.io import read_image
from torch.utils.data import Dataset




class trainImageDataset(Dataset):
    """
    changed from pytorch document:CustomImageDataset
    
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = np.array(self.img_labels.iloc[idx, 5:11].astype(int)) 
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)



"""
Data Augmentation: Resize to 224, Normalization (pretrained ResNet50 on ImageNet Dataset)
"""
augs = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),   
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])




class MyModel(nn.Module):
    """
    6 parallel output fc layers, each for 1 attri classification 
    """
    def __init__(self, class_attr):
        super(MyModel, self).__init__()
        
        self.model_resnet = resnet50(pretrained=True)

        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()

        self.fc1 = nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[0])) 
        self.fc2 = nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[1]))
        self.fc3 = nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[2]))
        self.fc4 = nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[3])) 
        self.fc5 = nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[4]))
        self.fc6 = nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[5]))

    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        out5 = self.fc5(x)
        out6 = self.fc6(x)
        return [out1, out2, out3, out4, out5, out6]



def train_loop(dataloader, model ,optimizer):
    """
    Baseline, loss function: cross entropy
    """
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        
        loss= 0
        
        for i in range(6):
            loss_fn = nn.CrossEntropyLoss()
            
            loss_i = loss_fn(pred[i], y[:,i])
            loss+=loss_i

        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss_, current = loss.item(), batch * len(X)
            print(f"loss: {loss_:>7f}  [{current:>5d}/{size:>5d}]")
        
    return loss.item()


def acc(dataloader, model):
    """
    test accuracy 
    """
    model.eval() # set to eval mode

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = [0, 0, 0, 0, 0, 0]

    with torch.no_grad():
        for X, y in dataloader:
            X,y =X.to(device), y.to(device)
            pred = model(X)
            for i in range(6):
                loss_fn = nn.CrossEntropyLoss()
                test_loss+=loss_fn(pred[i], y[:,i]).item()
                correct[i] += (pred[i].argmax(1) == y[:,i]).type(torch.float).sum().item()

                
    test_loss /= num_batches
    correct = np.divide(correct, size)
    mean_correct= sum(correct) / 6
    # print(f"Test Error : m Per-class Acc:{100*(mean_correct):>0.1f}%, Avg loss: {test_loss:>0.4f} \n")
    return test_loss, round(100*(mean_correct),2)



## Work Directory inputs (run annotationfile.py first)

img_dir='./'
train_file='./split/train_merge.txt'
val_file='./split/val_merge.txt'

class_attr=[7,3,3,4,6,3]

## GPU
device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


## model
model= MyModel(class_attr)
model.to(device)


## baseline model paremeters
BS=128
 
lr_adjust=1

learning_rate = lr_adjust* 0.001  # initial_lr
print('learning_rate:', learning_rate)

epochs = 20


## Dataloader

#seed set
torch.manual_seed(17) 

training_data = trainImageDataset(train_file, img_dir, transform=augs)
validation_data = trainImageDataset(val_file, img_dir, transform=augs)

train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=BS, shuffle=True)


## Optimization:
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

label_smooth=False

## training
train_acc_epch=[]
val_acc_epch=[]
train_loss_epch=[]
val_loss_epch=[]

for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    
    model.train()

    train_loss= train_loop(train_dataloader, model, optimizer)

    train_loss, train_mean_correct = acc(train_dataloader, model)
    val_loss, val_mean_correct = acc(val_dataloader, model)

    print('val_loss:', val_loss)
    print('val_acc:', val_mean_correct)

    train_acc_epch.append(train_mean_correct)
    val_acc_epch.append(val_mean_correct)

    train_loss_epch.append(train_loss)
    val_loss_epch.append(val_loss)

    print(train_acc_epch)
    print(val_acc_epch)
    

## Log save (run Log.py first)
Log=pd.read_csv('Log.csv')
dic={'val_Acc':val_mean_correct, 
'BS':BS,
'LR':learning_rate,
'Epoch':epochs,
'Optimizer':'Adam',
'lossfn':'CE', 
'label_smooth':label_smooth,
'Regularization':'None',
'Augment':'Basic'} 

Log =Log.append(dic,ignore_index=True)        

Log.to_csv('Log.csv', index=False)

attempts=len(Log)

df = pd.DataFrame(val_acc_epch, columns=["acc"])
df.to_csv('./epochacc/attempts_'+str(attempts)+'.csv',index=False)
df = pd.DataFrame(val_loss_epch, columns=["loss"])
df.to_csv('./epochloss/attempts_'+str(attempts)+'.csv',index=False)



title='Attempts:'+str(attempts)

plot = plt.figure()
plt.plot(train_acc_epch, label='train_acc')
plt.plot(val_acc_epch, label='val_acc')
plt.title(title)
plt.xlabel('Epoch')
plt.xlim(0,20)
plt.ylim(50,100)
plt.xticks(np.arange(0, 20, 1))
plt.legend()
plt.show()
plt.savefig('./Log_png_folder/attempt_'+str(attempts)+'_acc.png')


plot = plt.figure()
plt.plot(train_loss_epch, label='train_loss')
plt.plot(val_loss_epch, label='val_loss')
plt.title(title)
plt.xlabel('Epoch')
plt.xlim(0,20)
plt.ylim(0,10)
plt.xticks(np.arange(0, 20, 1))
plt.legend()
plt.show()
plt.savefig('./Log_png_folder/attempt_'+str(attempts)+'_loss.png')


print("Done and Plot!")

torch.save(model,'./model_folder/attempt_'+str(attempts)+'.pth')