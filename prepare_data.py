import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
from torch.autograd import Variable
import pprint 
import util 
import json 
import torch.nn.functional as F


all_xray_df = pd.read_csv('./data/sample_labels.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('images*','*.png'))}

print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['Path'] = all_xray_df['Image Index'].map(all_image_paths.get)

### convert age to int
all_xray_df['Patient Age'] = all_xray_df['Patient Age'].map(lambda x: str(x)[:-1]).astype(int)
########delete other columns of  all_xray_df only includes 'Path' and 'Finding Labels'
all_xray_df=all_xray_df[['Path','Finding Labels']]


### cleaning data and create columns for each pathology #####################
all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
from itertools import chain
all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        all_xray_df[c_label] = all_xray_df['Finding Labels'].map(lambda finding: 1 if c_label in finding else 0)

print (all_xray_df.sample(3))


############# keep at least 100 cases
#print (all_labels)
#MIN_CASES = 100
#all_labels = [c_label for c_label in all_labels if all_xray_df[c_label].sum()>MIN_CASES]
#print('Clean Labels ({})'.format(len(all_labels)), 
#      [(c_label,int(all_xray_df[c_label].sum())) for c_label in all_labels])

###sample 3000 from data using origial ratio 
sample_weights = all_xray_df['Finding Labels'].map(lambda x: len(x.split('|')) if len(x)>0 else 0).values + 4e-2
sample_weights /= sample_weights.sum()
all_xray_df = all_xray_df.sample(3000, weights=sample_weights)

###prepare training data
#all_xray_df['disease_vec'] = all_xray_df.apply(lambda x: [x[all_labels].values], 1).map(lambda x: x[0])
from sklearn.model_selection import train_test_split
####remove 'Finding Labels' colmn 
result_df=all_xray_df.drop(columns=['Finding Labels'])
train_df, temp_df = train_test_split(result_df,
                                   test_size = 0.4,
                                   random_state = 2018,
                                   stratify = all_xray_df['Finding Labels'].map(lambda x: x[:4]))
valid_df, test_df = train_test_split(temp_df,
                                   test_size = 0.5,
                                   random_state = 1846)
print('train', train_df.shape[0], 'validation', valid_df.shape[0], 'test', test_df.shape[0])


train_df.to_csv('./train.csv',index=False,header=True)
valid_df.to_csv('./valid.csv',index=False,header=True)
test_df.to_csv('./test.csv',index=False,header=True)

