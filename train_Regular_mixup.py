
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


augs = transforms.Compose([transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



class trainImageDataset(Dataset):
    def __init__(self, annotations_file,  annotations_file_shuffled,  img_dir, transform=None):
        self.img_labels_1 = pd.read_csv(annotations_file)
        self.img_labels_2 = pd.read_csv(annotations_file_shuffled)
        
        self.img_dir = img_dir
        
        self.transform = transform 
        

    def __len__(self):
        return len(self.img_labels_1)

    def __getitem__(self, idx):
        
        img_path_1 = os.path.join(self.img_dir, self.img_labels_1.iloc[idx, 0])
        img_path_2 = os.path.join(self.img_dir, self.img_labels_2.iloc[idx, 0])
        
        image_1 = read_image(img_path_1)
        image_2 = read_image(img_path_2)
        
        xstart=self.img_labels_1.iloc[idx,1]
        xend=self.img_labels_1.iloc[idx,3]
        ystart=self.img_labels_1.iloc[idx,2]
        yend=self.img_labels_1.iloc[idx,4]
        image_1= image_1[:, ystart:yend, xstart:xend]
        
        xstart=self.img_labels_2.iloc[idx,1]
        xend=self.img_labels_2.iloc[idx,3]
        ystart=self.img_labels_2.iloc[idx,2]
        yend=self.img_labels_2.iloc[idx,4]
        image_2= image_2[:, ystart:yend, xstart:xend]        
        
        
        _lambda = np.random.beta(0.2,0.2)
        
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
            image = _lambda*image_1   +  (1-_lambda)*image_2
        
        
        label_1 = torch.tensor(np.array(self.img_labels_1.iloc[idx, 5:11].astype(int)), dtype=torch.long)
        label_2 = torch.tensor(np.array(self.img_labels_2.iloc[idx, 5:11].astype(int)), dtype=torch.long)
        
        label=[]
        for i in range(6):
            label1 = torch.nn.functional.one_hot(label_1[i], class_attr[i])
            label2 = torch.nn.functional.one_hot(label_2[i], class_attr[i])
        
            label_i = _lambda*label1   +  (1-_lambda)*label2  
            
            label.append(label_i)
            
            
        return image, torch.tensor(np.array(label[0])), torch.tensor(np.array(label[1])),torch.tensor(np.array(label[2])), torch.tensor(np.array(label[3])),torch.tensor(np.array(label[4])), torch.tensor(np.array(label[5]))



class valImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform 

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)

        xstart=self.img_labels.iloc[idx,1]
        xend=self.img_labels.iloc[idx,3]
        ystart=self.img_labels.iloc[idx,2]
        yend=self.img_labels.iloc[idx,4]
        image= image[:, ystart:yend, xstart:xend]
        
        label = np.array(self.img_labels.iloc[idx, 5:11].astype(int)) 

        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)






class MyModel(nn.Module):
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


def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)


def mixup_train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y1,y2,y3,y4,y5,y6) in enumerate(dataloader):
        X, y1,y2,y3,y4,y5,y6=X.to(device), y1.to(device),y2.to(device),y3.to(device),y4.to(device),y5.to(device),y6.to(device)
        y=[y1,y2,y3,y4,y5,y6]
        
        # Compute prediction and loss
        pred = model(X)
        
        loss= 0
        for i in range(6):
            loss_i = cross_entropy_one_hot(pred[i], y[i])
            loss+=loss_i
    
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    return loss



def train_acc(dataloader, model):
    """
    """
    model.eval() # set to eval mode

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    correct = [0, 0, 0, 0, 0, 0]

    with torch.no_grad():
        for batch, (X, y1,y2,y3,y4,y5,y6) in enumerate(dataloader):
            X, y1,y2,y3,y4,y5,y6=X.to(device), y1.to(device),y2.to(device),y3.to(device),y4.to(device),y5.to(device),y6.to(device)
            y=[y1,y2,y3,y4,y5,y6]

            pred = model(X)
            for i in range(6):
                train_loss+=cross_entropy_one_hot(pred[i], y[i]).item()
                correct[i] += (pred[i].argmax(1) == y[i].argmax(1)).type(torch.float).sum().item()

                
    train_loss /= num_batches
    correct = np.divide(correct, size)
    mean_correct= sum(correct) / 6
    return train_loss, round(100*(mean_correct),2)


def test_acc(dataloader, model):
    """
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
    print(f"Test Error : Per-class Accuracy: {correct}, \n mAcc:{100*(mean_correct):>0.1f}%, Avg loss: {test_loss:>0.4f} \n")
    
    return test_loss, round(100*(mean_correct),2)







## Work Directory inputs
train_file='./split/train_merge.txt'
train_file_shuffled='./split/train_merge_shuffled.txt'
val_file='./split/val_merge.txt'
img_dir='./'

class_attr=[7,3,3,4,6,3]


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


## Optimizer
warmup=True
warm_up_epochs=3


## Dataloader
torch.manual_seed(17) 

training_data = trainImageDataset(train_file, train_file_shuffled, img_dir, transform=augs)
validation_data = valImageDataset(val_file, img_dir, transform=augs)


train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=BS, shuffle=True)

## Optimization:

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)

warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)


## Regularization:
label_smooth=False
# mix up


## training
train_acc_epch=[]
val_acc_epch=[]
train_loss_epch=[]
val_loss_epch=[]

for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    
    model.train()

    train_loss= mixup_train_loop(train_dataloader, model, optimizer)

    train_loss, train_mean_correct = train_acc(train_dataloader, model)
    val_loss, val_mean_correct = test_acc(val_dataloader, model)

    print('val_loss:', val_loss)
    print('val_acc:', val_mean_correct)

    train_acc_epch.append(train_mean_correct)
    val_acc_epch.append(val_mean_correct)

    train_loss_epch.append(train_loss)
    val_loss_epch.append(val_loss)

    print(train_acc_epch)
    print(val_acc_epch)
    
    scheduler.step()


## Training Log save
Log=pd.read_csv('Log.csv')
dic={'val_Acc':val_mean_correct, 
'BS':BS,
'LR':learning_rate,
'Epoch':epochs,
'Optimizer':'CosineLR+warmup+BBOX', 
'lossfn':'CE', # CE, weighted_CE, weighted_FocalLoss
'label_smooth':label_smooth,
'Regularization':'weightdecay:1e-5', # weightdecay
'Augment':'mixup'
} 


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
plt.savefig('./Log_png_folder/attempt_acc'+str(attempts)+'.png')


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
plt.savefig('./Log_png_folder/attempt_loss'+str(attempts)+'.png')


print("Done and Plot!")

torch.save(model,'./model_folder/attempt_'+str(attempts)+'.pth')

