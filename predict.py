import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import util
import torch.nn.functional as F
import torchvision.models as models
import time 
import os 
import json 


# convert dictionary to object
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def transform_data(data, use_gpu, train=False):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad=False)
    labels = Variable(labels, requires_grad=False)
    return inputs, labels


parser = util.get_parser()
parser.add_argument('--model_path',default=None,type=str,help="path to models")
local_args = parser.parse_args()

#print (local_args)
########################load parameter

print (os.path.join(local_args.model_path,'params.txt'))

params_dict = json.load(open(os.path.join(local_args.model_path,'params.txt'), 'r'))
folder_name = 'predictions/' + str(int(round(time.time() * 1000)))
args=Struct(**params_dict)

#print (args)


#########################test data set 

test_dataset = util.Dataset(args, "test")
test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=None)
########################load model 
use_gpu = torch.cuda.is_available()

nclasses=test_dataset.n_classes
if args.model == "densenet":
    model = DenseNet(args, nclasses)
elif args.model == "alexnet":
    model = models.__dict__[args.model](num_classes=nclasses)
elif args.model == "squeezenet":
    model = models.squeezenet1_1(num_classes=nclasses)
elif args.model == "inception":
    model = models.inception_v3(num_classes=nclasses)
elif args.model == "resnet":
    model = models.resnet50(num_classes=nclasses)
elif args.model == "vgg":
    model = models.vgg19(num_classes=nclasses)
else:
    print("{} is not a valid model.".format(args.model))


model.load_state_dict(torch.load(os.path.join(local_args.model_path,'val0.057482_train0.044659_epoch15')))
if use_gpu :
    model = model.cuda()

model.eval()
test_loss=0.0
test_counter=0
gts=[]
outs=[]
criterion = nn.MultiLabelSoftMarginLoss()
with torch.no_grad():
    for data in test_loader:
        inputs, labels = transform_data(data,use_gpu)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += (loss.item() * inputs.size(0))
        test_counter += inputs.size(0)
        out = torch.sigmoid(outputs).data.cpu().numpy()
        outs.extend(out)
        for gt in data[1].numpy().tolist():
            gts.append(gt)
outs=np.array(outs)
gts=np.array(gts)
print("Validation Loss: {:.6f}".format(test_loss/test_counter))

util.evaluate(gts, outs, test_loader.dataset.pathologies)


if not os.path.exists(folder_name):
    os.makedirs(folder_name)
name='prediction_probs'
np.save(folder_name + '/' + name, outs)
print("Predictions saved to ", folder_name)




