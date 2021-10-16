
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



class BBox_trainImageDataset(Dataset):
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


augs = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),   
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


class MyModel(nn.Module):
    def __init__(self, class_attr):
        super(MyModel, self).__init__()
        
        self.model_resnet = resnet50()

        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()

        self.fc1 = nn.Sequential(nn.Linear(num_ftrs, class_attr[0])) 
        self.fc2 = nn.Sequential(nn.Linear(num_ftrs, class_attr[1]))
        self.fc3 = nn.Sequential(nn.Linear(num_ftrs, class_attr[2]))
        self.fc4 = nn.Sequential(nn.Linear(num_ftrs, class_attr[3])) 
        self.fc5 = nn.Sequential(nn.Linear(num_ftrs, class_attr[4]))
        self.fc6 = nn.Sequential(nn.Linear(num_ftrs, class_attr[5]))

    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        out5 = self.fc5(x)
        out6 = self.fc6(x)
        return [out1, out2, out3, out4, out5, out6]




def train_loop(dataloader, model, lossfn ,optimizer):
    """
    """
    model.train()
    size = len(dataloader.dataset)
    img_labels = pd.read_csv('./split/train_merge.txt')

    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        
        loss= 0
        
        if lossfn=='CE':
            for i in range(6):
                loss_fn = nn.CrossEntropyLoss()
                loss+= loss_fn(pred[i], y[:,i])
            # loss_fn = nn.CrossEntropyLoss()
            # loss= loss_fn(pred[0], y[:,0])+loss_fn(pred[4], y[:,4])


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss_, current = loss.item(), batch * len(X)
            print(f"loss: {loss_:>7f}  [{current:>5d}/{size:>5d}]")
        
    return loss.item()


def fine_tuning(dataloader, model, lossfn ,optimizer):
    """
    """
    model.train()
    size = len(dataloader.dataset)
    img_labels = pd.read_csv('./split/train_merge.txt')

    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        
        loss= 0
        
        if lossfn=='CE':
            for i in range(6):
                loss_fn = nn.CrossEntropyLoss()
                loss+= loss_fn(pred[i], y[:,i])

        # Backpropagation        
        
        for name, param in model.named_parameters():
            if not 'fc' in name:
                param.requires_grad = False
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if batch % 5 == 0:
            loss_, current = loss.item(), batch * len(X)
            print(f"loss: {loss_:>7f}  [{current:>5d}/{size:>5d}]")
        
    return loss.item()





def acc(dataloader, model, lossfn):
    """
    """
    model.eval() # set to eval mode

    img_labels = pd.read_csv('./split/val_merge.txt')
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    correct = [0, 0, 0, 0, 0, 0]

    with torch.no_grad():
        for X, y in dataloader:
            X,y =X.to(device), y.to(device)
            pred = model(X)


            if lossfn=='CE':
                for i in range(6):
                    loss_fn = nn.CrossEntropyLoss()
                    loss_i = loss_fn(pred[i], y[:,i])
                    test_loss+=loss_i.item()
                    correct[i] += (pred[i].argmax(1) == y[:,i]).type(torch.float).sum().item()


    test_loss /= num_batches
    correct = np.divide(correct, size)
    mean_correct= sum(correct) / 6
    # print(f"Test Error : Per-class Accuracy: {correct}, \n mAcc:{100*(mean_correct):>0.1f}%, Avg loss: {test_loss:>0.4f} \n")
    return test_loss, round(100*(mean_correct),2)


torch.manual_seed(17) 

## Work Directory inputs
train_file='./split/train_merge.txt'
val_file='./split/val_merge.txt'
img_dir='./'


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

epochs = 10 #20

## Optimizer
warmup=True
warm_up_epochs=3


loss_function='CE' # 'CE', 'weighted_CE', 'weighted_FocalLoss'

BBox=True
label_smooth=False

## Dataloader

if BBox:
    training_data = BBox_trainImageDataset(train_file, img_dir, transform=augs)
    validation_data = BBox_trainImageDataset(val_file, img_dir, transform=augs)
else:
    training_data = trainImageDataset(train_file, img_dir, transform=augs)
    validation_data = trainImageDataset(val_file, img_dir, transform=augs)



train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=False)
val_dataloader = DataLoader(validation_data, batch_size=BS, shuffle=False)


## Optimization:
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

## Scheduler 
if not warmup:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)

elif warmup:
    warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)




## training

val_acc_epch=[]
val_loss_epch=[]


torch.manual_seed(17)
torch.cuda.manual_seed(17)
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False



# for t in range(epochs):
    
#     print(f"Epoch {t+1}\n-------------------------------")
    
#     model.train()

#     val_loss, val_mean_correct = acc(val_dataloader, model, loss_function)

#     train_loss= train_loop(train_dataloader, model, loss_function , optimizer)

#     # train_loss, train_mean_correct = acc(train_dataloader, model, loss_function)
#     val_loss, val_mean_correct = acc(val_dataloader, model, loss_function)

#     print('val_loss:', val_loss)
#     print('val_acc:', val_mean_correct)

#     val_acc_epch.append(val_mean_correct)

#     val_loss_epch.append(val_loss)

#     print(val_acc_epch)
    
#     scheduler.step()



##
for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    
    model.train()

    val_loss, val_mean_correct = acc(val_dataloader, model, loss_function)

    train_loss= fine_tuning(train_dataloader, model, loss_function , optimizer)

    val_loss, val_mean_correct = acc(val_dataloader, model, loss_function)

    print('val_loss:', val_loss)
    print('val_acc:', val_mean_correct)

    val_acc_epch.append(val_mean_correct)

    val_loss_epch.append(val_loss)

    print(val_acc_epch)
    
    scheduler.step()






# ## Training Log save
# Log=pd.read_csv('Log.csv')
# dic={'val_Acc':val_mean_correct, 
# 'BS':BS,
# 'LR':learning_rate,
# 'Epoch':epochs,
# 'Optimizer':'BBOX+CosineLR+warmup', #  'BBOX', 'BBOX+CosineLR+warmup'
# 'lossfn':loss_function,
# 'label_smooth':label_smooth,
# 'Regularization':'None',
# 'Augment':'Basic'} 

# Log =Log.append(dic,ignore_index=True)        

# Log.to_csv('Log.csv', index=False)

# attempts=len(Log)

# df = pd.DataFrame(val_acc_epch, columns=["acc"])
# df.to_csv('./epochacc/attempts_'+str(attempts)+'.csv',index=False)
# df = pd.DataFrame(val_loss_epch, columns=["loss"])
# df.to_csv('./epochloss/attempts_'+str(attempts)+'.csv',index=False)


# title='Attempts:'+str(attempts)

# plot = plt.figure()
# plt.plot(train_acc_epch, label='train_acc')
# plt.plot(val_acc_epch, label='val_acc')
# plt.title(title)
# plt.xlabel('Epoch')
# plt.xlim(0,20)
# plt.ylim(50,100)
# plt.xticks(np.arange(0, 20, 1))
# plt.legend()
# plt.show()
# plt.savefig('./Log_png_folder/attempt_acc'+str(attempts)+'.png')


# plot = plt.figure()
# plt.plot(train_loss_epch, label='train_loss')
# plt.plot(val_loss_epch, label='val_loss')
# plt.title(title)
# plt.xlabel('Epoch')
# plt.xlim(0,20)
# plt.ylim(0,10)
# plt.xticks(np.arange(0, 20, 1))
# plt.legend()
# plt.show()
# plt.savefig('./Log_png_folder/attempt_loss'+str(attempts)+'.png')


# print("Done and Plot!")

# torch.save(model,'./model_folder/attempt_'+str(attempts)+'.pth')