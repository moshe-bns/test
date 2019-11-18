from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        return  self.fc2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def my_loss1(pred, true_proportion):
    final = torch.argmax(pred, dim=1, keepdim=True)
    pred_proportion = final.sum()/float(len(pred))
    return torch.mul(pred_proportion-true_proportion,pred_proportion-true_proportion)

def my_loss2(output, true_proportion):
    loss = torch.mean((torch.mean(output) - true_proportion)**2)
    return loss

def my_loss3(pred, true_proportion):
    probs = torch.softmax(pred, dim=1)
    pred_proportion = torch.mean(probs[:,1])
    return torch.mul(pred_proportion-true_proportion,pred_proportion-true_proportion)

def myCrossEntropyLoss(outputs, labels):
  batch_size = outputs.size()[0]            # batch_size
  outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
  outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
  return -torch.sum(outputs)/batch_size

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # true_proportion = float(sum(target==1).tolist())/len(target)
        # wanted_proportion = np.random.choice([0.7,0.4,0.2,0.3], 1)[0]
        # num_true = int(32*wanted_proportion)
        # num_false = 32-num_true
        # true_proportion = float(num_true)/32
        # true_idx = np.random.choice(np.where(target==1)[0], num_true)
        # false_idx = np.random.choice(np.where(target==0)[0], num_false)
        # all_idx = true_idx.tolist() + false_idx.tolist()
        # np.random.shuffle(all_idx)
        # data = data[all_idx]
        # target = target[all_idx]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = mse_loss(output, target.view(32,1))

        # if true_proportion>0.5:
        #     target = torch.ones(target.shape, dtype=target.dtype)
        # else:
        #     target = torch.zeros(target.shape, dtype=target.dtype)
        # loss = my_loss3(output, true_proportion)
        output = torch.softmax(output, dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, trueP: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), 0))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            true_proportion = float(sum(target == 1).tolist()) / len(target)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data_set =datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))

    zero_idx = np.where(train_data_set.targets==0)[0]
    # zero_idx = np.random.choice(zero_idx, 2000)
    ones_idx = np.where(train_data_set.targets==1)[0]
    idx_train =zero_idx.tolist()+ones_idx.tolist()
    np.random.shuffle(idx_train)
    # idx_train =((train_data_set.targets==0)|(train_data_set.targets==1))
    train_data_set.data = train_data_set.data[idx_train][:100]
    train_data_set.targets = train_data_set.targets[idx_train][:100]

    train_loader = torch.utils.data.DataLoader(train_data_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_data_set = datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    idx_test = ((test_data_set.targets == 0) | (test_data_set.targets == 1))
    test_data_set.data = test_data_set.data[idx_test]
    test_data_set.targets = test_data_set.targets[idx_test]

    test_loader = torch.utils.data.DataLoader(test_data_set,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()