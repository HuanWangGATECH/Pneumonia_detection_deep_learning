# Pneumonia_detection_deep_learning

## Introduction 

Chest X-ray is one of the most common examination techniques used in hospitals today. X-ray is critical for screening, diagnosis, and management of thoracic diseases, many of which are among the leading causes of mortality worldwide. Every year, over 2 billion radio graphs is produced worldwide. However, detecting diseases from reading X-rays is still a challenging task that requires human expertise and time-consuming. Therefore, computational systems using big data tools that could read radio graphs as effectively as radiologists could bring much benefits to the current health screening procedure. ⁠


Due to increasing computational powers of computers, huge advances have been seen in large X-ray datasets and deep learning algorithms. Wang et al. provides one of the largest publicly available chest x-ray datasets ChestX-ray-14 with disease labels along with a small subset with region-level annotations (bounding boxes) for evaluation [1] ⁠. In addition, Wang et al. also bench-marked different convolutional neural network architectures pretrained on ImageNet. This development of dataset has motivated numerous works on deep learning algorithms to detect X-ray images. Guan et. al. addresses this problem by proposing a three-branch attention guided convolution neural network (AG-CNN) [2] ⁠. Li et. al. presents a unified approach that simultaneously performs disease identification and localization through the same underlying model for all images [3] ⁠. A novel method termed as segmentation-based deep fusion network (SDFN), which leverages the higher-resolution information of local lung regions, is developed [4] ⁠. Kumar et. al. experiments a set of deep learning models and present a cascaded deep neural network that can diagnose all 14 pathologies better than the baseline[5] ⁠. Yao et. al. adopts Long-short Term Memory Networks (LSTM) model to treat the multi-label classification as sequence prediction with a fixed length [6] ⁠. Rajpurkar et. al. develope CheXNeXt, a convolutional neural network to concurrently detect the presence of 14 different pathologies and [7], [8] ⁠. CheXNeXt’s discriminative performance is compared to the performance of 9 radiologists and achieved radiologist-level performance on 11 pathologies.

In this current project, two major challenges in X-ray disease diagnosis are overcome. Since multiple diseases could exist in one radio graph, classification of X-ray disease is a problem of multi-label classification. To solve this problem, a 14-dimension vector of diseases is used to represent presence of disease in each graph. A weighted cross entropy loss is adopted to represent of loss between predicted label and true label. The second challenge is to archive the high accuracy/cost ratio due to limitation of time and computational resources. Four different CNN models are trained on sample data set. Performances of those models are then compared to select the best fit model. The selected model is then trained on the whole X-ray dataset to gain model parameters that has the most predicting power.


## How to run 

==================================files================================
./   
predict.py----------model predicting file
prepare_data.py-----data preprocess 
README  
train.py------------model training file   
util.py-------------utility file
Team74_ChestXray.pdf
Team74_ChestXray.ppt

./data:
BBox_List_2017.csv  
Data_Entry_2017.csv  
sample_labels.csv  
test.csv  
train.csv  
train_relabeled.csv-----data file from ChestXNet paper  
valid.csv  
valid_relabeled.csv-----data file from ChestXNet paper 

./run_dir:
params.txt--------------------------------parameters used in training best model   
val0.057482_train0.044659_epoch15---------saved best model  

=========================how to run=====================================

1.run prepare_data.py to get train.csv test.csv valid.csv data 

command: python prepare_data.py

2.cp train.csv,test.csv,valid.csv to ./data folder

command: train.csv test.csv valid.csv ./data/

3.run train.py to train CNN models

command: python train.py --save_path run_dir --model alexnet --batch_size 8 --horizontal_flip --epochs 10 --lr 0.0001 --train_weighted --valid_weighted --scale 256

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="alexnet", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--drop_rate", default=0.0, type=float)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=123456, type=int)
    parser.add_argument("--tag", default="", type=str)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--scale", default=224, type=int)
    parser.add_argument("--horizontal_flip", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--scratch", action="store_true")
    parser.add_argument("--train_weighted", action="store_true")
    parser.add_argument("--valid_weighted", action="store_true")
    parser.add_argument("--size", default=None, type=str)
    return parser

4. run predict.py  to test CNN models

commnad: python predict.py --model_path run_dir  
