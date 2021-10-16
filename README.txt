create work dir 

./epochloss
./epochacc
./Log_png_folder
./model_folder

>>python annotationfile.py
merge trian_attr with its img dir and boundingbox information, ]

and the shuffle the training dataset, store the copy, in order to apply mixup algorithm later.


>> python Log.py
log file store the training process. like whats the optim, regularizer, data agumentation, lr, weight decay ...

>>python train_Baseline.py


>>python train_Optim.py

change BBox=True to use boundingbox information 

warmup=true  to add cosine LR with warmup epoch =3 

change loss_function to 'weighted_CE' 

change loss_function to 'weighted_FocalLoss'


>>python train_Regular.py
## baseline model: + bbox+LRstep+warmup 

change weightdecay = 1e-5

add label_smoothing =True

, reset, then add dataaugmentation

>>python train_Regular_mixup.py
## baseline model: + bbox+LRstep+warmup 
add mix up