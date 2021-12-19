# HW2-Code-Data
手写汉字识别代码(PyTorch)

## MNIST数据集文件说明
./MNIST/Data/MNIST/raw/
- train-images-idx3-ubyte：训练集图像数据，包含 60000 个训练样本；
- train-labels-idx1-ubyte：训练集标签； 
- t10k-images-idx3-ubyte：测试集图像数据，包含 10000 个测试样本；
- t10k-labels-idx1-ubyte：测试集标签。

## 源代码文件说明
- ./MNIST/mnist_vis.py: MNIST数据集的读取和可视化程序示例
- ./MNIST/mnist_train.py: MNIST数据训练集
- ./HW2Dataset/hw2_train.py: 学生用手写文字识别训练集

xcopy .\HW2Dataset\Raw_Data\train_Raw_Data\1\*.png .\HW2Dataset\Data\train\ /Y