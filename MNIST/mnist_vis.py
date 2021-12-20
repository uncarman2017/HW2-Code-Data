from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 读取MNIST的训练集和测试集数据
train_dataset = datasets.MNIST(root='./Data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./Data/', train=False, transform=transforms.ToTensor())

print("Shape of training data  :", train_dataset.data.size())
print("Shape of training labels:", train_dataset.targets.size())
print("Shape of testing  data  :", test_dataset.data.size())
print("Shape of testing  labels:", test_dataset.targets.size())

plot_pos = [231, 232, 233, 234, 235, 236]
fig = plt.figure()
for idx in range(6):
    ax = fig.add_subplot(plot_pos[idx])
    ax.imshow(train_dataset.data[idx].reshape(28, 28), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Label:' + str(train_dataset.targets[idx].numpy()))

plt.show()
