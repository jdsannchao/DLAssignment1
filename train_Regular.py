
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


## baseline model: + bbox+LRstep+warmup 

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



## using dataugmentation

    """
    Data Augmentation:

    resize to 224,224
    random horizontal flip
    scale hue, brightness, contrast, saturation
    gaussian blur
    normalised using ImagenetDataset statistics

    """

train_augs = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),   
            transforms.RandomHorizontalFlip(p=0.9),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.GaussianBlur(kernel_size=7, sigma=1.0), # sigma=(k-1)/6, the length for 99 percentile of gaussian pdf is 6sigma
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    

test_augs = transforms.Compose([transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])



## labelsmooth 
class LabelSmoothCELoss(nn.Module):
    """
    label smoothing, as described in XXXXXXXX
    compare to standard Binary CrossEntropy Loss Function: -log(P_t), 
    the label after label smoothing is no longer [0,0,1] but [0.05,0.05,0.9] if eps=0.1
    then the Cross Entropy Loss will be -0.9*log(P_t) + 1/K * mean of Softmax on the output layer.
    """

    def __init__(self, weights=None):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(-1)
        
    def forward(self, pred, y, eps=0.2):
        pred = -self.log_softmax(pred)
        K=pred.size(1)
        y_onenot= nn.functional.one_hot(y, K).float()
           
        label_smooth = (1.0 - eps) * y_onenot + eps / K
        
        loss = pred * label_smooth
        loss = loss.sum(axis=1)

        return loss.mean()


def train_loop(dataloader, model, lable_smooth, optimizer):
    """
    """
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        
        loss= 0

        if lable_smooth:
            for i in range(6):
                # weights=img_labels[i+5].value_counts(normalize=True).astype(np.float32).sort_index()
                # weights=torch.tensor(weights)
                # weights=weights.to(device)
                # loss_fn = LabelSmoothCELoss(weights)
                loss_fn = LabelSmoothCELoss()
                loss_i = loss_fn(pred[i], y[:,i])
                loss+=loss_i

        
        else:
            for i in range(6):
                # weights=img_labels[i+5].value_counts(normalize=True).astype(np.float32).sort_index()
                # weights=torch.tensor(weights)
                # weights=weights.to(device)
                # loss_fn = nn.CrossEntropyLoss(weights)

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


def acc(dataloader, model, label_smooth):
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

            if label_smooth:
                for i in range(6):
                    # weights=img_labels[i+1].value_counts(normalize=True).astype(np.float32).sort_index()
                    # weights=torch.tensor(weights)
                    # weights=weights.to(device)
                    # loss_fn = LabelSmoothCELoss(weights)

                    loss_fn=LabelSmoothCELoss()
                    test_loss+=loss_fn(pred[i], y[:,i]).item()
                    correct[i] += (pred[i].argmax(1) == y[:,i]).type(torch.float).sum().item()

            else:
                for i in range(6):
                    # weights=img_labels[i+5].value_counts(normalize=True).astype(np.float32).sort_index()
                    # weights=torch.tensor(weights)
                    # weights=weights.to(device)
                    # loss_fn = nn.CrossEntropyLoss(weights)

                    loss_fn = nn.CrossEntropyLoss()
                    test_loss+=loss_fn(pred[i], y[:,i]).item()
                    correct[i] += (pred[i].argmax(1) == y[:,i]).type(torch.float).sum().item()
                
    test_loss /= num_batches
    correct = np.divide(correct, size)
    mean_correct= sum(correct) / 6
    print(f"Test Error : Per-class Accuracy: {correct}, \n mAcc:{100*(mean_correct):>0.1f}%, Avg loss: {test_loss:>0.4f} \n")
    
    return test_loss, round(100*(mean_correct),2)


def cat_pred(dataloader, model, device):
    """

    """
    model.eval() 

    all_pred = torch.tensor([], device=device)
   
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            all_pred = torch.cat((all_pred, pred.argmax(1)), 0)

    return all_pred.cpu().numpy().astype(int) 




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

epochs = 20  


## Optimizer
warmup=True
warm_up_epochs=3

BBox=True

#Regularization
weightdecay=1e-5
label_smooth=False # False, True 


## Dataloader
torch.manual_seed(17) 

training_data = BBox_trainImageDataset(train_file, img_dir, transform=train_augs)
validation_data = BBox_trainImageDataset(val_file, img_dir, transform=test_augs)


train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=BS, shuffle=True)

## Regularization:
## weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weightdecay, amsgrad=False)
warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(epochs - warm_up_epochs) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)



## training
train_acc_epch=[]
val_acc_epch=[]
train_loss_epch=[]
val_loss_epch=[]

for t in range(epochs):
    
    print(f"Epoch {t+1}\n-------------------------------")
    
    model.train()

    train_loss= train_loop(train_dataloader, model, label_smooth, optimizer)

    train_loss, train_mean_correct = acc(train_dataloader, model, label_smooth)
    val_loss, val_mean_correct = acc(val_dataloader, model, label_smooth)

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
'Regularization':weightdecay, # weightdecay
'Augment':'Flip0.9, Colorjitter0.3, Blur[7x7, sigma=0.1]' #'Basic', 'Flip0.9, Colorjitter0.3, Blur[7x7, sigma=0.1]'
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