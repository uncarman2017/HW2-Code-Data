import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import itertools


# 载入MNIST数据集
def Load():
    # 载入训练集和测试集数据, MNIST方法中的参数从左到右分别是
    # root: 数据集文件根路径(规定为MNIST/processed/training.pt,test.pt),
    # train: True-从训练集取数据, False-从测试集取数据
    # transform: 一个函数，原始图片作为输入，返回一个转换后的图片
    # download: True-从互联网上下载数据集，并把数据集放在root目录下. 如果数据集之前下载过，将处理过的数据（minist.py中有相关函数）放在processed文件夹下
    train_dataset = datasets.MNIST(root='./Data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./Data/', train=False, transform=transforms.ToTensor())

    # 创建训练集和测试集数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。参数说明如下：
    # dataset: 加载数据的数据集
    # batch_size: 每个批处理加载多少个样本(默认: 1)
    # shuffle:  True-重新打乱样本数据(默认: False).
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# 构建一个卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 构建卷积层1,2,3
        # conv1~3: 使用Sequential类创建一个时序容器,容器中加入卷积核、激活函数ReLU以及最大池化层
        # Conv2d: 对由多个输入平面组成的输入信号应用二维卷积
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 10, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(10, 20, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(20, 40, 3), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        # 构建两个全连接层
        self.dense = torch.nn.Sequential(torch.nn.Linear(40, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10))

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return F.log_softmax(out, dim=1)


def update_confusion_matrix(predictions, labels, conf_matrix):
    for p, l in zip(predictions, labels):
        conf_matrix[l, p] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes, normalize, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    plt.axis("equal")
    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    plt.show()


def Train(epoch):
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = F.nll_loss(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] Accuracy: {}/{} ({:.1f}%) Loss: {:.6f}'.format(epoch,
                                                                                                  batch_idx * batch_size + len(
                                                                                                      data),
                                                                                                  len(train_loader.dataset),
                                                                                                  100. * (
                                                                                                          batch_idx + 1) / len(
                                                                                                      train_loader),
                                                                                                  correct,
                                                                                                  batch_idx * batch_size + len(
                                                                                                      data),
                                                                                                  100. * correct / (
                                                                                                          batch_idx * batch_size + len(
                                                                                                      data)),
                                                                                                  loss.item()), end="")

    print('\n')


def Test(epoch):
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
        print('\rTest Epoch: {} [{}/{} ({:.0f}%)] Accuracy: {}/{} ({:.1f}%) '.format(epoch,
                                                                                      batch_idx * batch_size + len(
                                                                                          data),
                                                                                      len(test_loader.dataset),
                                                                                      100. * (batch_idx + 1) / len(
                                                                                          test_loader), correct,
                                                                                      batch_idx * batch_size + len(
                                                                                          data), 100. * correct / (
                                                                                              batch_idx * batch_size + len(
                                                                                          data))), end="")
    print('\n')


def Output(conf_matrix_normalize):
    conf_matrix_train = torch.zeros(10, 10)
    conf_matrix_test = torch.zeros(10, 10)

    print('Outputting confusion matrix on training set ...\n')
    for batch_idx, (data, target) in enumerate(train_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        update_confusion_matrix(prediction, target, conf_matrix_train)

    plot_confusion_matrix(conf_matrix_train.numpy(), classes=range(10), normalize=conf_matrix_normalize,
                          title='Confusion matrix on training set')

    print('Outputting confusion matrix on testing set ...\n')
    for batch_idx, (data, target) in enumerate(test_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        update_confusion_matrix(prediction, target, conf_matrix_test)

    plot_confusion_matrix(conf_matrix_test.numpy(), classes=range(10), normalize=conf_matrix_normalize,
                          title='Confusion matrix on testing set')


if __name__ == "__main__":
    batch_size = 64  # 批处理记录数量
    epoch_num = 30  # 训练次数(轮数)

    train_loader, test_loader = Load()

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    for epoch in range(epoch_num):
        Train(epoch)
        Test(epoch)
    Output(conf_matrix_normalize=True)
