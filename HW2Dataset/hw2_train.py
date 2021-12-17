import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import itertools
import math
import cv2
import shutil
import os


class AIDataset(Dataset):
    def __init__(self, root, train, transform=None):
        self.transform = transform
        if train:
            self.data_dir = root + 'train/';
            self.file_list = os.listdir(self.data_dir)
            self.file_list.sort()
        else:
            self.data_dir = root + 'test/';
            self.file_list = os.listdir(self.data_dir)
            self.file_list.sort()

        img = cv2.imread(self.data_dir + self.file_list[0], 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = self.Preprocessing(img)

        self.dataset_len = len(self.file_list)
        self.data = np.empty((self.dataset_len, img.shape[0], img.shape[1]), dtype='float32')
        self.original_data = np.empty((self.dataset_len, 50, 50, 3), dtype='float32')
        self.targets = np.empty((self.dataset_len), dtype='int64')

        for i in range(self.dataset_len):
            img = cv2.imread(self.data_dir + self.file_list[i], 1)
            self.original_data[i] = img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = self.Preprocessing(img)
            self.data[i] = img
            self.targets[i] = int(self.file_list[i][0]) - 1

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        file_name = self.file_list[index]
        original_data = self.original_data[index]

        if self.transform:
            data = self.transform(data)
        return data, target, file_name, original_data

    def __len__(self):
        return self.dataset_len

    def Preprocessing(self, img):
        ########## Edit your code here ##########
        # Code for data preprocessing
        # Hints for preprocessing: filtering, normalization, cropping, scaling, ...

        ########## End your code here ##########
        return img


def Load():
    train_dataset = AIDataset(root='./Data/', train=True, transform=transforms.ToTensor())
    test_dataset = AIDataset(root='./Data/', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


########## Edit your code here ##########
# Code for building neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 9), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(882, 128), torch.nn.ReLU(), torch.nn.Linear(128, 9))

    def forward(self, x):
        conv1_out = self.conv1(x)
        res = conv1_out.view(conv1_out.size(0), -1)
        out = self.dense(res)
        return F.log_softmax(out, dim=1)


########## End your code here ##########

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
    for batch_idx, (data, target, _, _) in enumerate(train_loader):
        ########## Edit your code here ##########
        # Code for model training
        # Hints: forward propagation, loss function, back propagation, network parameter update, ...

        ########## End your code here ##########

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
    for batch_idx, (data, target, _, _) in enumerate(test_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

        print('\rTest  epoch: {} [{}/{} ({:.0f}%)] Accuracy: {}/{} ({:.1f}%) '.format(epoch,
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
    if not os.path.exists('./result'):
        os.mkdir('./result')
    else:
        shutil.rmtree('./result')
        os.mkdir('./result')
    os.mkdir('./result/train')
    os.mkdir('./result/test')
    for i in range(1, 10):
        file_dir = './result/train/' + str(i)
        os.mkdir(file_dir)
        os.mkdir(file_dir + '/True')
        os.mkdir(file_dir + '/False')
        file_dir = './result/test/' + str(i)
        os.mkdir(file_dir)
        os.mkdir(file_dir + '/True')
        os.mkdir(file_dir + '/False')

    conf_matrix_train = torch.zeros(9, 9)
    conf_matrix_test = torch.zeros(9, 9)

    print('Outputting confusion matrix on training set ...\n')
    for batch_idx, (data, target, file_list, original_data) in enumerate(train_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        update_confusion_matrix(prediction, target, conf_matrix_train)

        for i in range(len(data)):
            if prediction[i][0] == target[i]:
                cv2.imwrite('result/train/' + str(prediction[i][0].numpy() + 1) + '/True/' + file_list[i],
                            original_data[i].numpy())
            else:
                cv2.imwrite('result/train/' + str(prediction[i][0].numpy() + 1) + '/False/' + file_list[i],
                            original_data[i].numpy())

    plot_confusion_matrix(conf_matrix_train.numpy(), classes=range(1, 10), normalize=conf_matrix_normalize,
                          title='Confusion matrix on training set')

    print('Outputting confusion matrix on testing set ...\n')
    for batch_idx, (data, target, file_list, original_data) in enumerate(test_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        update_confusion_matrix(prediction, target, conf_matrix_test)

        for i in range(len(data)):
            if prediction[i][0] == target[i]:
                cv2.imwrite('result/test/' + str(prediction[i][0].numpy() + 1) + '/True/' + file_list[i],
                            original_data[i].numpy())
            else:
                cv2.imwrite('result/test/' + str(prediction[i][0].numpy() + 1) + '/False/' + file_list[i],
                            original_data[i].numpy())

    plot_confusion_matrix(conf_matrix_test.numpy(), classes=range(1, 10), normalize=conf_matrix_normalize,
                          title='Confusion matrix on testing set')


if __name__ == "__main__":
    model = Net()
    ########## Edit your code here ##########
    # Hyper-parameter adjustment and optimizer initialization
    # Information about optimizer in PyTorch: https://pytorch.org/docs/stable/optim.html & https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
    batch_size = 10
    epoch_num = 10
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    conf_matrix_normalize = True
    ########## End your code here ##########

    train_loader, test_loader = Load()
    for epoch in range(epoch_num):
        Train(epoch)
        Test(epoch)
    Output(conf_matrix_normalize=conf_matrix_normalize)
