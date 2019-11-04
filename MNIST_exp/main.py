import torch
from torch.utils.data import Dataset
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

class MNISTDataset(Dataset):

	def __init__(self,csv_file):
		self.data = pd.read_csv(csv_file, header=None)
		self.label = self.data[:][0]
		self.data = self.data[:][1:]


	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data

class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.fc1 = nn.Linear(28*28,28*28)
		self.fc2 = nn.Linear(28*28,28*28)
		self.fc3 = nn.Linear(28*28, 10)
		self.softmax = nn.Softmax()

	def forward(self,x):
		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)
		return self.softmax(x)





train = MNISTDataset('/home/ocelaiwo/Downloads/MNIST/mnist_train.csv')
test = MNISTDataset('/home/ocelaiwo/Downloads/MNIST/mnist_test.csv')



print(train[3])