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

train_loader = None
test_loader = None
model = None
optimizer = None
optimizer_2 = None


# 数据集对象构造器
# 参数:
# DataSet: 父类对象,pytorch的DataSet类型
class AIDataset(Dataset):
    # 类初始化方法,相当于构造函数,参数说明如下
    # self: 对象本身,构造函数必填参数且必须是第一个
    # root: 训练集文件所在根路径
    # train: true-训练集 false-测试集
    # tranform: 转换器函数，原始图片作为输入，返回一个转换后的图片
    def __init__(self, root, train, transform=None):
        self.transform = transform
        if train:
            self.data_dir = root + 'train/';
            self.file_list = os.listdir(self.data_dir)  # 存放数据集文件列表
            self.file_list.sort()
        else:
            self.data_dir = root + 'test/';
            self.file_list = os.listdir(self.data_dir)
            self.file_list.sort()
        # 从文件载入一个图片并返回一个包含图像数据的矩阵对象(读取失败则返回空矩阵), 第二个参数flag设为1(IMREAD_COLOR)表示始终将图片转换为RGB格式
        # imread返回对象的类型是ndarray(numpy中的类型)
        # img = cv2.imread(self.data_dir + self.file_list[0], cv2.IMREAD_COLOR)
        # # 将图片转换成灰度格式,增强识别效果
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # # 进一步进行图像的预处理
        # img = self.Preprocessing(img)
        # TODO: 笔者觉得以上三行代码冗余，下面有遍历数据集样本进行处理的相同代码

        # 读取样本集图片文件的数量作为数据集的长度
        self.dataset_len = len(self.file_list)
        # 创建一个空的矩阵,用于存放图片数据,empty方法第一个参数用于描述矩阵结构(用元组类型表示,包含数据集长度,单个图片长度和宽度),第二个参数为数据类型
        # TODO: 笔者不太明白为什么取第一个样本的长宽作为data对象中样本的shape, 第一个样本尺寸不标准咋办(虽然在执行当前训练代码之前,笔者有用清洗代码去除尺寸不符要求的样本,但感觉这样的代码逻辑不严密)
        # self.data = np.empty((self.dataset_len, img.shape[0], img.shape[1]), dtype='float32')
        self.data = np.empty((self.dataset_len, 50, 50), dtype='float32')
        # 创建一个空的矩阵，用于存放原始数据，图像结构规定为长*宽=50*50,RGB三原色
        self.original_data = np.empty((self.dataset_len, 50, 50, 3), dtype='float32')
        # 创建一个空的矩阵，用于存放每个样本的目标数据(真实数据)
        self.targets = np.empty((self.dataset_len), dtype='int64')

        # 遍历样本集进行处理
        for i in range(self.dataset_len):
            try:
                # 同上
                img = cv2.imread(self.data_dir + self.file_list[i], cv2.IMREAD_COLOR)
                self.original_data[i] = img
                # add by Max Yu 2021.12.19
                # if self.original_data[i].shape[0] != 50 or self.original_data[i].shape[1] != 50:
                #     self.original_data[i] = self.data[i] = self.targets[i] = None
                #     print("The image %s size is not correct" %(self.file_list[i]))
                #     continue

                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = self.Preprocessing(img)
                self.data[i] = img
                # TODO: 文件名第一个字节就是图像对应的数字,即目标值,为啥-1,似乎和后面的损失值计算有关
                self.targets[i] = int(self.file_list[i][0]) - 1
            except Exception as err:  # Add by Max Yu 2021.12.19
                print(err)
                continue

    # pytorch的DataSet子类必须重写的方法, 用于获取数据集中的行
    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        file_name = self.file_list[index]
        original_data = self.original_data[index]

        if self.transform:
            data = self.transform(data)
        return data, target, file_name, original_data

    # pytorch的DataSet子类必须重写的方法, 用于获取数据集中的长度
    def __len__(self):
        return self.dataset_len

    # 样本集预处理
    def Preprocessing(self, img):
        ########## Edit your code here ##########
        # Code for data preprocessing
        # Hints for preprocessing: filtering, normalization, cropping, scaling, ...
        # add by Max Yu 2021.12.19
        #
        if img.shape[0] > 50 or img.shape[1] > 50:  # 缩图用INTER_AREA算法较好
            img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_AREA)
        elif img.shape[0] < 50 and img.shape[1] < 50:  # 扩图用INTER_LINEAR算法较快(INTER_CUBIC算法较慢)
            img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_LINEAR)
        ########## End your code here ###########
        return img


# 载入数据集
def load():
    # 载入训练集和测试集数据, MNIST方法中的参数从左到右分别是
    # root: 数据集文件根路径(规定为MNIST/processed/training.pt,test.pt),
    # train: True-从训练集取数据, False-从测试集取数据
    # transform: 一个函数，原始图片作为输入，返回一个转换后的图片
    train_dataset = AIDataset(root='./Data/', train=True, transform=transforms.ToTensor())
    test_dataset = AIDataset(root='./Data/', train=False, transform=transforms.ToTensor())

    _train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    _test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return _train_loader, _test_loader


########## Edit your code here ##########
# 神经网络模型对象，从torch.nn.Module基类继承
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 构建卷积层1,2,3
        # conv1~3: 使用Sequential类创建一个时序容器,容器中加入卷积核、激活函数ReLU以及最大池化层
        # Conv2d(): 对由多个输入平面组成的输入信号应用二维卷积,参数说明: in_channels 为输入信号的通道数；out_channels 为输出信号的通道数；kernel_size 为卷积核的尺寸
        # MaxPool2d(): 用于进行最大池化操作,参数说明: kernel_size 为最大池化的窗口大小
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=9), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(kernel_size=2))
        # add by Max Yu 2021.12.19
        self.conv2 = torch.nn.Sequential(torch.nn.Conv2d(2, 196, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2))
        self.conv3 = torch.nn.Sequential(torch.nn.Conv2d(196, 882, 3), torch.nn.Dropout2d(0.7), torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(2))
        # 构建两个全连接层
        # Linear(): 对输入进行线性变换，可用于构建全连接层, 参数说明: in_features 为输入的维度；out_features 为输出的维度；
        self.dense = torch.nn.Sequential(torch.nn.Linear(in_features=882, out_features=128), torch.nn.ReLU(),
                                         torch.nn.Linear(128, 9))

    # 重载基类方法,连接各层,构建向前传播网络链路
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv1_out.view(conv3_out.size(0), -1)
        # 第一个全连接层的激活函数同样为 ReLU，最后一层为 log_softmax
        out = self.dense(res)
        return F.log_softmax(out, dim=1)  # dim参数为log_softmax层的维度


########## End your code here ##########


def update_confusion_matrix(predictions, labels, conf_matrix):
    for p, l in zip(predictions, labels):
        conf_matrix[l, p] += 1
    return conf_matrix


# 输出混淆矩阵图表
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


def train(epoch_index):
    correct = 0
    for batch_idx, (data, target, _, _) in enumerate(train_loader):
        ########## Edit your code here ##########
        # Code for model training
        # Hints: forward propagation, loss function, back propagation, network parameter update, ...
        # add by Max Yu 2021.12.19
        _model = model(data)
        loss = F.nll_loss(_model, target)

        optimizer_2.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_2.step()

        ########## End your code here ##########
        # 读取每列最大的值作为预测值
        prediction = _model.data.max(1, keepdim=True)[1]
        # 判断预测值和目标值是否一致,计算识别准确的数量
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()
        # 输出的参数分别是: 样本记录流水号, 已处理样本数/总的样本数, 完成率, 正确识别的样本数/样本总数, 准确率, 损失量
        print('\rTrain Epoch: {} [{}/{} ({:.0f}%)] Accuracy: {}/{} ({:.1f}%) Loss: {:.6f}'
              .format(epoch_index, batch_idx * batch_size + len(data), len(train_loader.dataset),
                      100. * (batch_idx + 1) / len(train_loader), correct, batch_idx * batch_size + len(data),
                      100. * correct / (batch_idx * batch_size + len(data)), loss.item()), end="")
    print('\n')


def test(epoch_index):
    correct = 0
    for batch_idx, (data, target, _, _) in enumerate(test_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        prediction = output.data.max(1, keepdim=True)[1]
        correct += prediction.eq(target.data.view_as(prediction)).cpu().sum()

        print('\rTest Epoch: {} [{}/{} ({:.0f}%)] Accuracy: {}/{} ({:.1f}%) '
              .format(epoch_index, batch_idx * batch_size + len(data), len(test_loader.dataset),
                      100. * (batch_idx + 1) / len(test_loader), correct, batch_idx * batch_size + len(data),
                      100. * correct / (batch_idx * batch_size + len(data))), end="")
    print('\n')


# 输出结果
def output(conf_matrix_normalize):
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

    # 构建9*9矩阵并用0填充
    conf_matrix_train = torch.zeros(9, 9)
    conf_matrix_test = torch.zeros(9, 9)

    print('Outputting confusion matrix on training set ...\n')
    for batch_idx, (data, target, file_list, original_data) in enumerate(train_loader):
        with torch.no_grad():
            data, target = Variable(data), Variable(target)  # 最近版本的pytorch中,Variable函数已经废弃
        _model = model(data)
        prediction = _model.data.max(1, keepdim=True)[1]
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
        _model = model(data)
        prediction = _model.data.max(1, keepdim=True)[1]
        update_confusion_matrix(prediction, target, conf_matrix_test)

        for i in range(len(data)):
            if prediction[i][0] == target[i]:
                cv2.imwrite('result/test/' + str(prediction[i][0].numpy() + 1) + '/True/' + file_list[i],
                            original_data[i].numpy())
            else:
                cv2.imwrite('result/test/' + str(prediction[i][0].numpy() + 1) + '/False/' + file_list[i],
                            original_data[i].numpy())
    # 绘制图表
    plot_confusion_matrix(conf_matrix_test.numpy(), classes=range(1, 10), normalize=conf_matrix_normalize,
                          title='Confusion matrix on testing set')


if __name__ == "__main__":
    model = Net()
    ########## Edit your code here ##########
    # Hyper-parameter adjustment and optimizer initialization
    # Information about optimizer in PyTorch: https://pytorch.org/docs/stable/optim.html &
    # https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-optim/
    # MDF by Max Yu 2021.12.19 以下代码调节模型超参数
    batch_size = 64
    epoch_num = 128
    # MDF by Max Yu 2021.12.19 加入两个优化器可以提高识别精度
    # SGD函数实现随机梯度下降算法, 参数说明如下:
    # params: 待优化参数的iterable或者是定义了参数组的dict;
    # lr: 学习率
    # momentum：可选,动量因子,默认0
    # weight_decay: 权重衰减(L2惩罚), 默认0
    optimizer = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.5)
    optimizer_2 = optim.SGD(params=model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.5)
    ########## End your code here ##########

    train_loader, test_loader = load()
    for epoch in range(epoch_num):
        train(epoch)
        test(epoch)
    output(conf_matrix_normalize=True)
