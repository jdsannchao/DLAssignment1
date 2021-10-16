
# import model.model_R18_R50_Tw
# from model import FocalLoss

#######testseettsetset########

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
    def __init__(self, annotations_file, img_dir, _nclass, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None, sep=' ')
        self.img_dir = img_dir
        self._nclass = _nclass
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, self._nclass] # 1:6
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        return image, label



class testImageDataset(Dataset):
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



def train_trans(aug=True):
    if aug:
        shape_aug = transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2))
        color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

        train_augs = transforms.Compose([transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),shape_aug,color_aug,
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    else:
        train_augs=transforms.Compose([transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()])

    return train_augs


def test_trans():
    test_augs = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),
        transforms.ToTensor()])

    return test_augs


def choose_mode(arch, nclass):

    if arch=='ResNet18':

        model = resnet18(pretrained=True)

        model.fc = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(True),
        nn.Linear(1024, nclass)) 
        return model

    elif arch=='ResNet50':

        model = resnet50(pretrained=True)

        model.fc = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.ReLU(True),
        nn.Linear(1024, nclass)
        )
        return model       


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X,y =X.to(device), y.to(device)
        
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        scheduler.step()

def test_acc(dataloader, model, loss_fn, nclass):
    """
    """
    model.eval() # set to eval mode

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X,y =X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error for class {nclass}: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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

train_file='./split/train_merge_attr.txt'
test_file='./split/val_merge_attr.txt'
val_file='./split/test.txt'
img_dir='./'
class_attr=[7,3,3,4,6,3]

learning_rate = 0.1 #initial_lr
epochs = 10  #10 

arch='ResNet50'
trans=False
loss='BCE'
FL_gamma=2 # tuning from [0.5,1,1.5,2]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


img_labels = pd.read_csv('./split/train_merge_attr.txt',header=None,sep=' ')


torch.manual_seed(17) # Data augmentation involves randomness.

test_accuracy=[]
pred_dict={}

for i in range(1,7):
    training_data = trainImageDataset(train_file, img_dir,_nclass=i, transform=train_trans(trans))
    testing_data = trainImageDataset(test_file, img_dir,_nclass=i, transform=test_trans())

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=64, shuffle=True)

    weights=img_labels[i].value_counts(normalize=True).astype(np.float32).sort_index()

    weights=torch.tensor(weights)

    weights=weights.to(device)
    
    model=choose_mode(arch,class_attr[i-1])
    model.to(device)


    if loss=='Focal':
        loss_fn=FocalLoss.FocalLoss(alpha=weights, gamma=FL_gamma)
    elif loss=='BCE' :
        loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)


    model.train() # set to train mode

    for t in range(epochs):

        print(f"Class{i} - Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)

    acc=test_acc(test_dataloader, model, loss_fn, i)
    test_accuracy.append(acc)

    ## for prediction prurpose
    val_data = testImageDataset(val_file, img_dir, transform=test_trans())
    val_dataloader = DataLoader(val_data, batch_size=64, shuffle=False)

    all_pred =cat_pred(val_dataloader, model, device)
    
    pred_dict[i] = all_pred


   
print(test_accuracy)

# df = pd.concat([pd.Series(v, name=k) for k, v in pred_dict.items()], axis=1)
# print(df.head(5))    
# # df.to_csv('./split/test_pred.txt',header=False, index=False)


