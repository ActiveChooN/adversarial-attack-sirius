import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from pathlib import Path
import requests
import sys


class MNISTDataset(Dataset):

	def __init__(self, csv_file):
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
		self.fc1 = nn.Linear(28*28, 28*28)
		self.fc2 = nn.Linear(28*28, 20)
		self.fc3 = nn.Linear(10, 10)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x1 = x[:, :10]
		x2 = x[:, 10:]
		x2 = F.relu(self.fc3(x2))
		return torch.cat((x1, x2), 1)


class NetExact10(nn.Module):

	def __init__(self):
		super(NetExact10, self).__init__()
		self.fc1 = nn.Linear(28*28, 28*28)
		self.fc2 = nn.Linear(28*28, 10)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x


def cross_entropy_transferring(output, target):
	loss1 = F.cross_entropy(output[:, :10], target)
	loss2 = F.cross_entropy(output[:, 10:], target)
	return loss1 - loss2


def train_adversarial(n_epochs, transferring_lr, fc3_lr, logging=True):
	optimizer_transferring = torch.optim.SGD(set(net.parameters()) - set(net.fc3.parameters()), lr=transferring_lr)
	optimizer_fc3 = torch.optim.SGD(net.fc3.parameters(), lr=fc3_lr)
	for epoch in range(n_epochs):
		entries_processed = 0
		for batch in dataloader:
			optimizer_transferring.zero_grad()
			optimizer_fc3.zero_grad()
			output = net(batch['image'])
			loss_ce_transfer = cross_entropy_transferring(output, batch['label'])
			loss_f3 = F.cross_entropy(output[:, 10:], batch['label'])
			loss_ce_transfer.backward(retain_graph=True)
			loss_f3.backward()
			optimizer_transferring.step()
			optimizer_fc3.step()
			entries_processed += 1
			if (entries_processed % 3000 == 0) and logging:
					print(f"Processed {entries_processed * 4} entries")
		if logging:
			print(f"Epoch {epoch} complete")

def train_fc3(n_epochs, fc3_lr, logging=True):
	optimizer_fc3 = torch.optim.SGD(net.fc3.parameters(), lr=fc3_lr)
	for epoch in range(n_epochs):
		processed_entries = 0
		for batch in dataloader:
			optimizer_fc3.zero_grad()
			output = net(batch['image'])
			loss_f3 = F.cross_entropy(output[:, 10:], batch['label'])
			loss_f3.backward()
			optimizer_fc3.step()
			processed_entries += 1
			if (processed_entries % 3000 == 0) and logging:
				print(f"Processed {processed_entries * 4} entries")
		if logging:
			print(f"Epoch {epoch} complete")

def evaluate_results(steps = -1):
	dataloader_test = DataLoader(test,batch_size=4, shuffle=True)
	count_first10 = 0
	count_last10 = 0
	total = 0
	for i, batch in enumerate(dataloader_test):
		raw_result = net(batch['image'])
		probs_first10 = F.softmax(raw_result[:,:10], dim = 1)
		probs_last10 = F.softmax(raw_result[:,10:], dim = 1)
		result_first10 = torch.argmax(probs_first10, dim = 1)
		result_last10 = torch.argmax(probs_last10, dim = 1)
		label = batch['label']
		for j in range(4):
			if(result_first10[j] == label[j]):
				count_first10 += 1
			if(result_last10[j] == label[j]):
				count_last10 += 1
			total += 1
		if i == steps-1:
			break
	print("Accuracy")
	print(f"Softmax on first 10 components: {count_first10/total}")
	print(f"Softmax on last 10 components: {count_last10/total}")

if __name__ == '__main__':
	if not os.path.exists(Path('mnist_train.csv')):
		if len(sys.argv) < 2 or sys.argv[1] != 'download':
			print("No train data")
			exit(1)
		print("Downloading train data")
		url = "https://pjreddie.com/media/files/mnist_train.csv"
		r = requests.get(url)
		with open(Path('mnist_train.csv'),'wb') as f:
			f.write(r.content)
	if not os.path.exists(Path('mnist_test.csv')):
		if len(sys.argv) < 2 or sys.argv[1] != 'download':
			print("No test data")
			exit(1)
		print("Downloading test data")
		url = "https://pjreddie.com/media/files/mnist_test.csv"
		r = requests.get(url)
		with open(Path('mnist_test.csv'), 'wb') as f:
			f.write(r.content)
	train = MNISTDataset(Path('mnist_train.csv'))
	test = MNISTDataset('mnist_test.csv')
	dataloader = DataLoader(train, batch_size=4, shuffle=True)
	net = Net()
	train_adversarial(5,0.01,0.01)
	evaluate_results()
	print("Teaching Fully Connected 3")
	train_fc3(5,0.01)
	evaluate_results()