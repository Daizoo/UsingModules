# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# %%
# 入力データをpytorch用に変換するため
# ToTensorはtorch.tensorに変換する用、Normalizeは入力のデータをそれぞれ設定した
# 平均と分散に正規化する
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# 今回の学習に使う画像データセットであるCIFAR10をダウンロードする
# train=trueとすると学習用に使うフラグを建てるらしい
# transformにはデータセットを入力用に変換するためのメソッドを書く、今回は上記で設定したものを
# 使う。ちなみに、今回は設定していないが入力時のサンプリング方法も指定することが可能
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
# 学習に使うジェネレーターを作成。batch_sizeを指定することで一度に取得するデータ量を指定できる
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# こちらは評価用データセット。やることは同じ
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
# クラス一覧
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%

# 画像を表示するための関数
# 非正規化(半分にして0.5を足す)をすれば基の画像データに戻る


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# イテレータ化して適当にデータを取り出す
dataiter = iter(trainloader)
images, labels = dataiter.next()

# 画像を表示する
imshow(torchvision.utils.make_grid(images))
# 付随するラベルも表示する
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# %%

# ネットワーク構築
# やっていることは"Neural Networks"と同じ


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
# %%
# 損失関数と最適化手法を指定
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# %%
# ネットワークの学習

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
# %%
# ネットワークを保存
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
# %%
# 実際にデータを出力させてみる
# まずは正解データを表示
dataiter = iter(testloader)
images, labels = dataiter.next()

# 画像を表示
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# %%
# ネットワーク再ロード
net = Net()
net.load_state_dict(torch.load(PATH))
# %%
# ネットワークに通して結果を出させる
outputs = net(images)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))
# %%
# 正答率を計測してみる
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
# %%
# クラス(ラベル)ごとに正答率を計測
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
# %%
# GPU計算できるか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# %%
# ここから先は計算を行うときにGPU側に値を渡すための方法が書かれているが、
# そんな難しいことでもないので本家を読んでください
