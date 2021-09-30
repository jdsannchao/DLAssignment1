
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.models import resnet18
from torchvision.models import resnet50

import pandas as pd
import numpy as np

import os
from torchvision.io import read_image
from torch.utils.data import Dataset

class FocalLoss(torch.nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    """
 
    def __init__(self, alpha = None, gamma: float = 0., reduction: str = 'mean', ignore_index: int = 100000):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper. Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.Defaults to 'mean'.
        """
 
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
 
        self.log_softmax = torch.nn.LogSoftmax(-1) # dim=-1

        self.nll_loss = torch.nn.NLLLoss(weight=alpha, reduction='none', ignore_index=ignore_index)
 
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        
        # place for label smoothing:
        
        # https://blog.csdn.net/u013066730/article/details/94545407
            
        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = self.log_softmax(x)
        ce = self.nll_loss(log_p, y)
 
        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]
 
        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma
 
        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce
 
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
 
        return loss


class trainImageDataset(Dataset):
    """
    
    ## Comment !!!!!!!!!!!!!!
    
    """
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, sep=' ')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = np.array(self.img_labels.iloc[idx, 1:7].astype(int)) 
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


class predImageDataset(Dataset):
    """
    
    ## Comment !!!!!!!!!!!!!!
    
    """

    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, sep=' ')
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image



def train_trans(augment=True):
    if augment:
        train_augs = transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),   
            transforms.RandomHorizontalFlip(p=0.9),
            # transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    else:
        train_augs=transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    return train_augs


def test_trans():
    test_augs = transforms.Compose([transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    return test_augs

class MyModel(nn.Module):
    def __init__(self, arch, class_attr):
        super(MyModel, self).__init__()

        if arch=='ResNet18':

            self.model_resnet = resnet18(pretrained=True)

        elif arch=='ResNet50':

            self.model_resnet = resnet50(pretrained=True)

        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 =nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[0])) 
        self.fc2 = nn.Linear(num_ftrs, class_attr[1])
        self.fc3 = nn.Linear(num_ftrs, class_attr[2])
        self.fc4 =nn.Sequential(nn.Linear(num_ftrs, 1024),nn.ReLU(True),
                                nn.Linear(1024, class_attr[3])) 
        self.fc5 = nn.Linear(num_ftrs, class_attr[4])
        self.fc6 = nn.Linear(num_ftrs, class_attr[5])

    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        out3 = self.fc3(x)
        out4 = self.fc4(x)
        out5 = self.fc5(x)
        out6 = self.fc6(x)
        return out1, out2, out3, out4, out5, out6


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)
        #print(y[0])
        # Compute prediction and loss
        pred1,pred2,pred3,pred4,pred5,pred6 = model(X)
        # print(pred1.type)
        # y=y.unsqueeze(1)
        loss = 2*loss_fn(pred1, y[:,0])+loss_fn(pred2, y[:,1])+loss_fn(pred3, y[:,2])+2*loss_fn(pred4, y[:,3])+loss_fn(pred5, y[:,4])+loss_fn(pred6, y[:,5])
        # loss = loss_fn(pred1, y[:,0])
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        # scheduler.step()



def acc(dataloader, model, loss_fn):
    """

    """
    model.eval() # set to eval mode

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct1, correct2,correct3,correct4,correct5,correct6,= 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y =X.to(device), y.to(device)
            pred1,pred2,pred3,pred4,pred5,pred6 = model(X)
            # test_loss += loss_fn(pred1, y[:,0])
            test_loss += loss_fn(pred1, y[:,0]).item()+loss_fn(pred2, y[:,1]).item()+loss_fn(pred3, y[:,2]).item()+loss_fn(pred4, y[:,3]).item()+loss_fn(pred5, y[:,4]).item()+loss_fn(pred6, y[:,5]).item()
            correct1 += (pred1.argmax(1) == y[:,0]).type(torch.float).sum().item()
            correct2 += (pred2.argmax(1) == y[:,1]).type(torch.float).sum().item()
            correct3 += (pred3.argmax(1) == y[:,2]).type(torch.float).sum().item()
            correct4 += (pred4.argmax(1) == y[:,3]).type(torch.float).sum().item()
            correct5 += (pred5.argmax(1) == y[:,4]).type(torch.float).sum().item()
            correct6 += (pred6.argmax(1) == y[:,5]).type(torch.float).sum().item()
            correct=[correct1,correct2,correct3,correct4,correct5,correct6]

    test_loss /= num_batches
    correct = np.divide(correct, size)
    print(f"Test Error : \n Accuracy: {correct}, Avg loss: {test_loss:>8f} \n")
    return correct



def cat_pred(dataloader, model, device):
    """


    """
    model.eval() # set to eval mode

    all_pred = torch.tensor([], device=device)
   
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = model(X)
            all_pred = torch.cat((all_pred, pred.argmax(1)), 0)

    return all_pred.cpu().numpy().astype(int) 


## Work Directory inputs
img_labels = pd.read_csv('./split/train_merge_attr.txt',header=None,sep=' ')

train_file='./split/train_merge_attr.txt'
val_file='./split/val_merge_attr.txt'
img_dir='./'
pred_file='./split/test.txt'

class_attr=[7,3,3,4,6,3]


## model paremeters
BS=64 # Batch Size

# learning_rate = 0.1* 0.1 * BS/256 # initial_lr, linear scaling

learning_rate = 0.1   # 和0.1的比 小四倍直接就不train了... 哎 

epochs = 20  #10,20,30  

arch='ResNet18' # 'ResNet50' AWS上，本地只能测试18(就这还老没空间...)

augment=False # base line 是会有的 测试好了这里就改成真的MixUp了

loss='CE' # 先用的CE， 真的要测有 BCE, FocalLoss

FL_gamma=2 # 这个是Focal Loss 的 hyper, 记得在paper里提选了一个最好的。tuning from [0.5,1,1.5,2]


## GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

## model

model=MyModel('ResNet18', class_attr)
model.to(device)

# torch.manual_seed(17) # Data augmentation involves randomness.


## weight = weights 
# weights=img_labels[1].value_counts(normalize=True).astype(np.float32).sort_index()
# weights=torch.tensor(weights)
# weights=weights.to(device)

## Dataloader

training_data = trainImageDataset(train_file, img_dir, transform=train_trans(augment))
validation_data = trainImageDataset(val_file, img_dir,transform=test_trans())

train_dataloader = DataLoader(training_data, batch_size=BS, shuffle=True)
val_dataloader = DataLoader(validation_data, batch_size=BS, shuffle=True)

# loss_fn = nn.CrossEntropyLoss(weight=weights) # 这里应该调成一个可以修改的模式。回头再说....
loss_fn = nn.CrossEntropyLoss() 
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) #例子是0.1 但总觉得那样太小根本不够...


## training
model.train()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    
print("Done!")

acc=acc(val_dataloader, model, loss_fn)
