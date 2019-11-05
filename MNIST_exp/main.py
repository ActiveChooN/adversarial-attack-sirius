import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

class MNISTDataset(Dataset):

	def __init__(self,csv_file):
		self.data = pd.read_csv(csv_file, header=None)
		self.label = self.data[:][0]
		self.data = self.data.drop([0], axis=1) / 255


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		image = self.data.iloc[idx].values.astype(np.float32)
		cls = self.label.iloc[idx]
		return {'image': image, 'label': cls}

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(28*28,28*28)
		self.fc2 = nn.Linear(28*28,20)
		self.fc3 = nn.Linear(10,10)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x[:,10:] = F.relu(self.fc3(x[:,10:]))
		return x

class NetExact10(nn.Module):

	def __init__(self):
		super(NetExact10, self).__init__()
		self.fc1 = nn.Linear(28*28,28*28)
		self.fc2 = nn.Linear(28*28,10)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

def cross_entropy_transferring(output, target):
	loss1 = F.cross_entropy(output[:,:10],target)
	loss2 = F.cross_entropy(output[:,10:],target)
	return loss1 - loss2

train = MNISTDataset('/home/ocelaiwo/Downloads/MNIST/mnist_train.csv')
test = MNISTDataset('/home/ocelaiwo/Downloads/MNIST/mnist_test.csv')
dataloader = DataLoader(train, batch_size=4, shuffle=True)
netExact10 = NetExact10()
optimizerExact10 = torch.optim.SGD(netExact10.parameters(), lr=0.01)

# rr = 0
# for batch in dataloader:
# 	optimizerExact10.zero_grad()
# 	output = netExact10(batch['image'])
# 	loss = F.cross_entropy(output,batch['label'])
# 	loss.backward()
# 	optimizerExact10.step()
# 	print(loss)
# 	rr += 1
# 	if(rr > 1):
# 		break

# a = nn.Softmax(dim = 0)
# print(a(netExact10(torch.tensor(test[4]['image']))))
# print(test[4]['label'])

net = Net()
optimizer_transferring = torch.optim.SGD(set(net.parameters()) - set(net.fc3.parameters()),lr=0.01)
optimizer_fc3 = torch.optim.SGD(net.fc3.parameters(), lr=0.01)

rr = 0
for batch in dataloader:
	optimizer_transferring.zero_grad()
	optimizer_fc3.zero_grad()
	output = net(batch['image'])
	loss_ce_transfer = cross_entropy_transferring(output,batch['label'])
	loss_f3 = F.cross_entropy(output[:,10:],batch['label'])
	loss_ce_transfer.backward()
	loss_f3.backward()
	optimizer_transferring.step()
	optimizer_fc3.step()
	print("------------------------------")
	print(loss_f3)
	print(loss_ce_transfer)
	print("------------------------------")
	rr += 1
	if(rr > 1000):
		break
