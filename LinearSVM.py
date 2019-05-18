import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torch.utils.data import DataLoader

class LinerSVM(nn.Module):
	def __init__(self,num_class=10):
		super(LinerSVM,self).__init__()
		self.hidden_size = num_class
		self.fc1 = nn.Linear(784,256)
		self.fc2 = nn.Linear(256,num_class)
		# torch.nn.init.kaiming_uniform(self.fc1.weight)
		# torch.nn.init.constant(self.fc1.bias,0.1)
		# torch.nn.init.kaiming_uniform(self.fc2.weight)
		# torch.nn.init.constant(self.fc2.bias,0.1)

	def forward(self,input):
		x = self.fc1(input)
		x = self.fc2(x)
		return x


class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight#weight for each class, size=n_class, variable containing FloatTensor,cuda,reqiures_grad=False
        self.size_average=size_average
    def forward(self, output, y):#output: batchsize*n_class
        #print(output.requires_grad)
        #print(y.requires_grad)
        output_y=output[torch.arange(0,y.size()[0]).long(),y.data].view(-1,1)#view for transpose
        #margin - output[y] + output[i]
        loss=output-output_y+self.margin#contains i=y
        #remove i=y items
        loss[torch.arange(0,y.size()[0]).long(),y.data]=0
        #max(0,_)
        loss[loss<0]=0
        #^p
        if(self.p!=1):
            loss=torch.pow(loss,self.p)
        #add weight
        if(self.weight is not None):
            loss=loss*self.weight
        #sum up
        loss=torch.sum(loss)
        if(self.size_average):
            loss/=output.size()[0]#output.size()[0]
        return loss

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


data_tf = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5], [0.5]) # 标准化
])


num_epoch = 10
batch_size = 64

train_set = MNIST('C:/Users/Jet Zhang/Desktop/pytorch/mnist', train=True, transform=data_tf)
test_set = MNIST('C:/Users/Jet Zhang/Desktop/pytorch/mnist', train=False, transform=data_tf)

train_data = DataLoader(train_set, batch_size , True, num_workers=0)
test_data = DataLoader(test_set, batch_size*2 , False, num_workers=0)


net = LinerSVM(10)

optimzier = torch.optim.Adadelta(net.parameters(), 1e-3)

criterion = multiClassHingeLoss()

for epoch in range(num_epoch):
	train_loss = 0
	train_acc = 0
	count = 0
	net = net.train()
	for im, label in train_data:
		count += 1
		im = Variable(im.view(-1,784))
		label = Variable(label)
		output = net(im)
		loss = criterion(output,label)
		# loss = torch.mean(torch.clamp(1 - output.t() * label.float(), min=0))  # hinge loss
		loss += 0.1 * (torch.mean(net.fc1.weight ** 2)+torch.mean(net.fc2.weight ** 2))  # l2 penalty

		optimzier.zero_grad()
		loss.backward()
		optimzier.step()

		train_loss += loss.item()
		train_acc += get_acc(output, label)

		# if count % 100 == 0:
		# 	print("{} loss is: {}".format(count,loss.item()))

	if test_data is not None:
		valid_loss = 0
		valid_acc = 0
		net = net.eval()
		for im, label in test_data:
			im = Variable(im.view(-1,784))
			label = Variable(label)
			output = net(im)
			loss = criterion(output,label)
			# loss = torch.mean(torch.clamp(1 - output.t() * label.float(), min=0))  # hinge loss
			valid_loss += loss.item()
			valid_acc += get_acc(output, label)
		print("Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
				% (epoch, train_loss / len(train_data),
				train_acc / len(train_data), valid_loss / len(test_data),
				valid_acc / len(test_data)))
	else:
		print("Epoch %d. Train Loss: %f, Train Acc: %f, " %
				(epoch, train_loss / len(train_data),
				train_acc / len(train_data)))


'''
LinearSVM:
	Epoch 1. Train Loss: 2.353030, Train Acc: 0.769273, Valid Loss: 1.692794, Valid Acc: 0.805479, 
	Epoch 2. Train Loss: 1.515052, Train Acc: 0.818197, Valid Loss: 1.242025, Valid Acc: 0.840981, 
	Epoch 3. Train Loss: 1.191244, Train Acc: 0.840185, Valid Loss: 1.026180, Valid Acc: 0.858386, 
	Epoch 4. Train Loss: 1.016612, Train Acc: 0.854361, Valid Loss: 0.897680, Valid Acc: 0.867385, 
	Epoch 5. Train Loss: 0.906001, Train Acc: 0.863739, Valid Loss: 0.813502, Valid Acc: 0.873220, 
	Epoch 6. Train Loss: 0.829505, Train Acc: 0.870586, Valid Loss: 0.752789, Valid Acc: 0.882516, 
	Epoch 7. Train Loss: 0.773327, Train Acc: 0.875150, Valid Loss: 0.705706, Valid Acc: 0.885779, 
	Epoch 8. Train Loss: 0.729414, Train Acc: 0.879931, Valid Loss: 0.669048, Valid Acc: 0.887955, 
	Epoch 9. Train Loss: 0.694844, Train Acc: 0.883612, Valid Loss: 0.639960, Valid Acc: 0.891218, 
	
	
'''