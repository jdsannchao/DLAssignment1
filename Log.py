import pandas as pd

col_list=['val_Acc','BS','LR','Epoch','Optimizer','lossfn','label_smooth','Regularization','Augment']

df = pd.DataFrame(columns=col_list)

df.to_csv('Log.csv', index=False)

