import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython

class FCMetaNet(nn.Module):

    def __init__(self, first_layer_size):
        super(FCMetaNet, self).__init__()
        self.fc1 = nn.Linear(first_layer_size, 4096) #input dimension both output and fc layer output
        self.bn1 = nn.BatchNorm1d(4096)

        self.fc2 = nn.Linear(4096, 7000)
        self.bn2 = nn.BatchNorm1d(7000)
        self.fc3 = nn.Linear(7000, 2048)
        self.bn3 = nn.BatchNorm1d(2048)

        self.fc4 = nn.Linear(2048, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc5 = nn.Linear(512, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc6 = nn.Linear(512, 64)
        self.bn6 = nn.BatchNorm1d(64)

        self.fc7 = nn.Linear(64, 2)
        self._relu = nn.ReLU()
        self._dropout = nn.Dropout()


    def forward(self, x):
        x = self._relu(self.fc1(x))
        x = self.bn1(x)
        #x = F.dropout(x, training=self.training)
        x = self._relu(self.fc2(x))
        x = self.bn2(x)

        x = self._dropout(x)

        x = self._relu(self.fc3(x))
        x = self.bn3(x)

        #x = F.dropout(x, training=self.training)
        x = self._relu(self.fc4(x))
        x = self.bn4(x)
        x = self._dropout(x)

        x = self._relu(self.fc5(x))
        x = self.bn5(x)
        #x = F.dropout(x, training=self.training)

        x = self._relu(self.fc6(x))
        x = self.bn6(x)

        x = self.fc7(x)

        return F.log_softmax(x, dim = 1)

def train_meta(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        #only need to put tensors in position 0 onto device (?)
        inputs = torch.Tensor(len(data), data[0].shape[0], data[0].shape[1])
        data[0] = data[0].to(device)
        torch.cat(data, out=inputs)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * data[0].shape[0], len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    print('percentage correct:')
    correct_percent = 100.0*correct/len(train_loader.dataset)
    print(correct_percent)
    return correct_percent

def test_meta_model(model, device, error_test_loader, correct_test_loader, optimizer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    accuracies = []
    with torch.no_grad():
        test_loss = 0
        correct = 0
        correct_acc = 0
        error_acc = 0

        for batch_idx, (data, target) in enumerate(correct_test_loader):
            #only need to put tensors in position 0 onto device (?)
            inputs = torch.Tensor(len(data), data[0].shape[0], data[0].shape[1])
            data[0] = data[0].to(device)
            torch.cat(data, out=inputs)
            #IPython.embed()
            target = target.to(device)
            output = model(inputs)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


        test_loss /= len(correct_test_loader.dataset)
        print('\nTest set: Average loss on correctly classified examples: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(correct_test_loader.dataset),
            100. * correct / len(correct_test_loader.dataset)))
        correct_acc = 100. * correct / len(correct_test_loader.dataset)

        test_loss = 0
        correct = 0

        for batch_idx, (data, target) in enumerate(error_test_loader):
            #only need to put tensors in position 0 onto device (?)
            inputs = torch.Tensor(len(data), data[0].shape[0], data[0].shape[1])
            data[0] = data[0].to(device)
            torch.cat(data, out=inputs)

            target = target.to(device)
            output = model(inputs)
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


        test_loss /= len(error_test_loader.dataset)
        print('\nTest set: Average loss on incorrectly classified examples: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(error_test_loader.dataset),
            100. * correct / len(error_test_loader.dataset)))
        error_acc = 100. * correct / len(error_test_loader.dataset)
    return (correct_acc, error_acc)
