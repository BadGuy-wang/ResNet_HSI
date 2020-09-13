import os
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch as t
from tqdm import tqdm


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    '''
    实现ResNet34，ResNet34包含多个layer，每个layer又包含多个残差模块
    利用_make_layer实现layer
    '''

    def __init__(self, inputchannel, num_classes):
        super(ResNet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=inputchannel, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(128, 256, 3)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)

        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.view(200, 1, 1)
        x = x.float()
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.max_pool2d(x, 3, 1, 1)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train(net, optimizer, loss_function, data_loader, val_loader, epoch, device):
    net.to(device)
    save_epoch = epoch // 10 if epoch > 20 else 1
    train_loss = []
    val_accuracies = []
    for e in tqdm(range(0, epoch)):
        net.train()
        for batch_idx, (data, label) in enumerate(data_loader):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            # print('train_batch_idx', batch_idx)
            out = net(data)
            loss = loss_function(out, label.long())

            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_acc = val(net, val_loader, ignored_labels=[0], device=device)
        val_accuracies.append(val_acc)
        metric = abs(val_acc)

        if e % save_epoch == 0:
            save_model(net, 'ResNet_run' + str(e) + '_' + str(metric), 'checkpionts/')
    return train_loss, val_accuracies


def val(net, val_loader, ignored_labels, device):
    net.to(device)
    accuracy, total = 0., 0.
    net.eval()
    for batch_idx, (data, target) in enumerate(val_loader):
        # print('val_batch_idx', batch_idx)
        # Load the data into the GPU if required
        data, target = data.to(device), target.to(device)

        output = net(data)

        _, output = t.max(output, dim=1)

        for out, pred in zip(output.view(-1), target.view(-1)):
            if out.item() in ignored_labels:
                continue
            else:
                accuracy += out.item() == pred.item()
                total += 1

    net.train()
    return accuracy / total


def test(net, data_loader, device):
    net.to(device)
    net.eval()
    pred_labels = []
    for batch_idx, (data, _) in enumerate(data_loader):
        with t.no_grad():
            data = data.to(device)
            out = net(data)
            _, output = t.max(out, dim=1)
            # print(output)
            pred_labels.append(output)
    return pred_labels


def save_model(model, model_name, save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    if isinstance(model, t.nn.Module):
        t.save(model.state_dict(), save_dir + model_name + '.pth')
    else:
        print('Model is error')
