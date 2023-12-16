# My-own-Self-Distillation-Model
This is my own implement of Self-Distillation Model for practicing purpose.  \
I have constructed the whole model.  And this is the **version 1.0**.  \
If you have any questions,please contact with me. Glad to make friends, who are interesting in CV or KD as well.  \
My e-mail address: 10235101553@stu.ecnu.edu.cn \
\
The original paper is here.  \
Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation  \
<https://arxiv.org/abs/1905.08094> 

## How to run the code？ 
First, create “model” and "dataset" folder on your local environment, which are used for saving checkpoints and dataset relatively. \
Second, run **dataset.py** to download cifar100 dataset. \
Third, run **model.py** to initialize the original Resnet18 backbone model. \
Final, run **train.py** to construct the whole model and train it. 

## Some details 
In the first 100 batch, the loss will grow conversely. It's normal, don't worry.  \
You can try various hyper-parameters on your own to get better results. \
Try to change: \
    1. T in loss.py  T means distillation temperature.\
    2. lamda in loss.py  lamba is the balance factor to balance hard-label loss and soft-label loss. 
