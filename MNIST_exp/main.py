import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

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
            output = net(batch[0])
            loss_ce_transfer = cross_entropy_transferring(output, batch[1])
            loss_f3 = F.cross_entropy(output[:, 10:], batch[1])
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
            output = net(batch[0])
            loss_f3 = F.cross_entropy(output[:, 10:], batch[1])
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
        raw_result = net(batch[0])
        probs_first10 = F.softmax(raw_result[:,:10], dim = 1)
        probs_last10 = F.softmax(raw_result[:,10:], dim = 1)
        result_first10 = torch.argmax(probs_first10, dim = 1)
        result_last10 = torch.argmax(probs_last10, dim = 1)
        label = batch[1]
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

    from torchvision import transforms
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.view(-1))])
    train = torchvision.datasets.MNIST(root='./datasets', train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(root='./datasets',train=False,download=True, transform=transform)
    dataloader = DataLoader(train, batch_size=4, shuffle=True)
    net = Net()
    train_adversarial(5,0.01,0.01)
    evaluate_results()
    print("Teaching Fully Connected 3")
    train_fc3(5,0.01)
    evaluate_results()