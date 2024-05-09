import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

class Net(torch.nn.Module):
    def __init__(self): #定义网络结构
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)   #Linear层，输入特征数28*28，输出特征数64
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)  #Linear层，输入特征数64，输出特征数10，10个output对应10个类别，是数字0-9

    def forward(self, x):   #定义前向传播过程，x表示输入
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

def get_data_loader(is_train):  #定义数据加载器，is_train表示是否是训练集
    to_tensor = transforms.Compose([transforms.ToTensor()]) #将数据转换为Tensor
    data_set = MNIST("./datasets", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True) #一个批次包含15张图片，shuffle=True每次数据集打乱

def evaluate(test_data, net):  #评估神经网络识别准确率，输入为测试集和神经网络
    n_correct = 0   #预测正确的数目
    n_total = 0     #总数目
    with torch.no_grad():
        for (x, y) in test_data:
            output = net.forward(x.view(-1, 28*28))     #输入x，进行前向传播，得到预测值output
            for i, output in enumerate(output):
                if torch.argmax(output) == y[i]:        #argmax返回每一行中最大值元素的索引，即预测值
                    n_correct += 1                      #判断预测值与实际值是否相等，相等则n_correct加1
                n_total += 1
    return n_correct / n_total

def train(train_data, net):
    print("initial accuracy: ", evaluate(test_data, net)) #打印初始准确率
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001) #定义优化器
    for epoch in range(2):
        for (x, y) in train_data:
            net.zero_grad() #初始化，梯度清零
            output = net.forward(x.view(-1, 28*28)) #输入x，进行前向传播，得到预测值output
            loss = torch.nn.functional.nll_loss(output, y) #计算损失，nll_loss表示负对数似然
            loss.backward() #反向传播
            optimizer.step() #更新参数
        print("epoch: ", epoch, "accuracy: ", evaluate(test_data, net)) #打印每个epoch的准确率


def predict(test_data, net):
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()

if __name__ == "__main__":    
    if torch.cuda.is_available():
        device = torch.device("cuda")  # 使用GPU
        print("GPU is available.")
    else:
        device = torch.device("cpu")  # 使用CPU
        print("GPU is not available, using CPU.")

    train_data = get_data_loader(True)   #加载训练集
    test_data = get_data_loader(False)    #加载测试集
    net = Net()                           #定义神经网络    

    train(train_data, net)
    predict(test_data, net)