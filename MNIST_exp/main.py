import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


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

# class Net(nn.Module):
#
# 	def __init__(self):
# 		super(Net, self).__init__()
# 		self.fc1 = nn.Linear(28*28,28*28)
# 		self.fc2 = nn.Linear(28*28,20)
#
# 	def forward(self,x):
# 		x = self.fc1(x)
# 		x = self.fc2(x)
# 		return x

class NetDirect10(nn.Module):

	def __init__(self):
		super(NetDirect10, self).__init__()
		self.fc1 = nn.Linear(28*28,28*28)
		self.fc2 = nn.Linear(28*28,10)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

train = MNISTDataset('/home/ocelaiwo/Downloads/MNIST/mnist_train.csv')
test = MNISTDataset('/home/ocelaiwo/Downloads/MNIST/mnist_test.csv')
dataloader = DataLoader(train, batch_size=4, shuffle=True)
net = NetDirect10()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

rr = 0
for batch in dataloader:
	optimizer.zero_grad()
	output = net(batch['image'])
	loss = criterion(output,batch['label'])
	loss.backward()
	optimizer.step()
	print(loss)
	rr += 1
	if(rr > 10000):
		break

a = nn.Softmax(dim = 0)
print(a(net(torch.tensor(test[4]['image']))))
print(test[4]['label'])
