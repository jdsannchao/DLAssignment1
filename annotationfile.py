import pandas as pd

img_dir='./'
train_labels='./split/train_attr.txt'
val_labels='./split/val_attr.txt'

train_bbox= './split/train_bbox.txt'
val_bbox= './split/val_bbox.txt'
test_bbox= './split/test_bbox.txt'

train_img='./split/train.txt'
val_img='./split/val.txt'
train_img='./split/test.txt'

_img= pd.read_csv('./split/train.txt', header=None, sep=' ')
_img.columns=['img_dir']
_bbox= pd.read_csv('./split/train_bbox.txt', header=None, sep=' ')
_bbox.columns=['xstart','ystart','xend','yend']
_labels= pd.read_csv('./split/train_attr.txt', header=None, sep=' ')
df=pd.concat([_img,_bbox,_labels], axis=1)
df.to_csv('./split/train_merge.txt', index=False)

df_shuffled =df.sample(frac=1).reset_index(drop=True)
df_shuffled.to_csv('./split/train_merge_shuffled.txt', index=False)


_img= pd.read_csv('./split/val.txt', header=None, sep=' ')
_img.columns=['img_dir']
_bbox= pd.read_csv('./split/val_bbox.txt', header=None, sep=' ')
_bbox.columns=['xstart','ystart','xend','yend']
_labels= pd.read_csv('./split/val_attr.txt', header=None, sep=' ')
df=pd.concat([_img,_bbox,_labels], axis=1)
df.to_csv('./split/val_merge.txt', index=False)

_img= pd.read_csv('./split/test.txt', header=None, sep=' ')
_img.columns=['img_dir']
_bbox= pd.read_csv('./split/test_bbox.txt', header=None, sep=' ')
_bbox.columns=['xstart','ystart','xend','yend']
df=pd.concat([_img,_bbox], axis=1)
df.to_csv('./split/test_merge.txt', index=False)