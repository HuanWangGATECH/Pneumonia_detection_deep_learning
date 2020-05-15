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
from torch.optim import lr_scheduler

class DenseNet(nn.Module):
    def __init__(self, config, nclasses):
        super(DenseNet, self).__init__()
        self.model_ft = densenet121(pretrained=not config.scratch, drop_rate=config.drop_rate)
        num_ftrs = self.model_ft.classifier.in_features
        self.model_ft.classifier = nn.Linear(num_ftrs, nclasses)
        self.config = config
    def forward(self, x):
        return self.model_ft(x)

def transform_data(data, use_gpu):
    inputs, labels = data
    labels = labels.type(torch.FloatTensor)
    if use_gpu is True:
        inputs = inputs.cuda()
        labels = labels.cuda()
    inputs = Variable(inputs, requires_grad=False, volatile=not train)
    labels = Variable(labels, requires_grad=False, volatile=not train)
    return inputs, labels

def get_loss(dataset, weighted):

    criterion = nn.MultiLabelSoftMarginLoss()

    def loss(preds, target):

        if weighted:

            return dataset.weighted_loss(preds, target)

        else:

            return criterion(preds, target)

    return loss




use_gpu = torch.cuda.is_available()
model = None
##find out if gpu is being used 
print ("current device used ",torch.cuda.get_device_name(torch.cuda.current_device()))


parser = util.get_parser()
args = parser.parse_args()
print (args)
#####saving parameters to params.txt file
if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)

with open(os.path.join(args.save_path, "params.txt"), 'w') as out:
    json.dump(vars(args), out, indent=4)


train, val = util.load_data(args) ###train and validation dataloader 
nclasses = train.dataset.n_classes
print("Number of classes:", nclasses)
#model=Net(nclasses)
if args.model == "densenet":
    model = DenseNet(args, nclasses)
elif args.model == "alexnet":
    model = models.__dict__[args.model](num_classes=nclasses)
elif args.model == "squeezenet":
    model = models.squeezenet1_1(num_classes=nclasses)
elif args.model == "resnet":
    model = models.resnet50(num_classes=nclasses)
elif args.model == "inception":
    model = models.inception_v3(num_classes=nclasses)
elif args.model == "vgg":
    model = models.vgg16(num_classes=nclasses)
else:
    print("{} is not a valid model.".format(args.model))


if use_gpu:
    model=model.cuda()

optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=args.lr,weight_decay=args.weight_decay)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.001, factor=0.1)
####get loss function 

train_criterion = get_loss(train.dataset, args.train_weighted)
val_criterion = get_loss(val.dataset, args.valid_weighted)

###############################training 
b_time=time.time()
epochs=args.epochs
train_losses = []
valid_losses = []
counter=0
best_model_wts, best_loss = model.state_dict(), float("inf")

data_time=[]
train_time=[]
val_time=[]
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_counter = 0
    for batch_idx, data in enumerate(train):
        #print ("=================")
        #print (np.shape(data[0]))
        #print (batch_idx,data)
        #inputs, labels = transform_data(data, True, train=True)
        inputs, labels = transform_data(data, use_gpu)
        e_time=time.time()-b_time
        #print ("############Data loading and transforming time",e_time)
        data_time.append(e_time)
        b_time=time.time()

        optimizer.zero_grad()
        outputs = model(inputs)
        #print ("=========labels==========")
        #print (labels.size())
        #print (labels)
        #print ("==========output==========")
        #print (outputs.size())
        #print (outputs)
        loss = train_criterion(outputs, labels)
        #print ("========lost data=========")
        #print (loss.data)
        loss.backward()
        optimizer.step()
        train_loss += (loss.item() * inputs.size(0))
        train_counter += inputs.size(0)
        e_time=time.time()-b_time
        train_time.append(e_time)
        #print ("############training 1 batch time",e_time)
        b_time=time.time()

    train_loss=train_loss/train_counter
    train_losses.append(train_loss)
    
    
    # switch to evaluation mode
    model.eval()
    b_time=time.time()
    valid_loss=0.0
    valid_counter=0
    outs=[]
    gts=[]
    with torch.no_grad():
        for batch_idx, data in enumerate(val):
            inputs, labels = transform_data(data,use_gpu)
            outputs = model(inputs)
            loss = val_criterion(outputs, labels)
            valid_loss += (loss.item() * inputs.size(0))
            valid_counter += inputs.size(0)
            out = torch.sigmoid(outputs).data.cpu().numpy()
            outs.extend(out)
            for gt in data[1].numpy().tolist():
                gts.append(gt)
            e_time=time.time()-b_time
            val_time.append(e_time)
            #print ("############training 1 batch time",e_time)
            b_time=time.time()
    outs=np.array(outs)
    gts=np.array(gts)
    valid_loss=valid_loss/valid_counter
    valid_losses.append(valid_loss)
    util.evaluate(gts, outs, val.dataset.pathologies)
    ####adjust learning rate or not 

    scheduler.step(valid_loss)
    ####compare valid_loss with best_loss and update best_loss
    if (valid_loss < best_loss):
        best_loss = valid_loss
        best_model_wts = model.state_dict()
        counter = 0
    else:
        counter += 1

    if counter > 10:
        break

    print("Epoch: {:d} Batch: {:d} ({:d}) Train Loss: {:.6f} Valid Loss: {:.6f}".format(
            epoch, batch_idx, args.batch_size,train_loss,valid_loss))
    print ("each batch of each epoch, data loading time is {},training time is {},validation time (data and output){}".format(np.average(data_time),np.average(train_time),np.average(val_time)))


    ######save model 
    torch.save(best_model_wts, os.path.join(args.save_path, "val%f_train%f_epoch%d" % (valid_loss, train_loss, epoch)))


###########finished training 
#######################plot learning curve 
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.plot(np.arange(len(train_losses)), train_losses, label='Train')
plt.plot(np.arange(len(valid_losses)), valid_losses, label='Validation')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(loc="best")
plt.savefig('learning_curve_val{:.6f}_train{:.6f}.png'.format(valid_loss,train_loss))



