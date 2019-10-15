# %%
# neural networls
import torch.optim as optim
import torch as t
import torch.nn as nn
import torch.nn.functional as F

# %%
# ネットワークのクラスを定義
# nn.Moduleを引き継ぐことで、torchの恩恵を受けることができる


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # Conv2dの引数: サンプル数、チャネル数、フィルタのサイズ
        # 今回はサンプル(入力画像枚数)が1、チャネル数(フィルタの数)が6
        # フィルタのサイズ(畳み込みの範囲)を3として畳み込み第1層を指定
        self.conv1 = nn.Conv2d(1, 6, 3)
        # 同様に畳み込み第2層をサンプル数6(前の層のチャネル数と同じ)、チャネル数を16
        # フィルタのサイズを3として構築する
        self.conv2 = nn.Conv2d(6, 16, 3)
        # 線形結合層 y=Wx+b の構築
        # 重みWで出力の次元を圧縮していく
        # Linearは入力サイズ、出力サイズを引数に取る
        # 畳み込み層2つ、サブサンプリングとしてMAXプーリングを使うので
        # 最終的な線形全結合の入力サイズは16*6*6らしい
        # 画像系ニューラルネットワークに詳しくはないのでなぜ最終出力がこれになるかは不明
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 順伝搬に関する処理を書いた関数
        # 第1層処理: 入力→活性化関数(ReLu) プーリング層(2*2)→出力
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # 第2層処理: 上と同じ
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 特徴量数から2次元の行列に調整
        x = x.view(-1, self.num_flatfeatures(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flatfeatures(self, x):
        # Conv2dの出力値には最初の要素にテンソルのデータを保持しているので
        # それを取り除いて保持する
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# 定義ここまで
# %%
net = Net()
print(net)
# %%
# netのパラメータを見てみる
params = list(net.parameters())
print(len(params))
# 第1層畳み込み
print(params[0].size())
# 第2層畳み込み
print(params[2].size())
# %%
# 順伝搬テスト
input = t.randn(1, 1, 32, 32)  # cov2dの入力情報として必要なサンプル数を付与して生成
out = net(input)
print(out)
# %%
target = t.randn(1, 10)
criterion = nn.MSELoss()  # 平均二乗誤差
loss = criterion(out, target)
print(loss)
# %%
# どの損失関数が使われているかわかる
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
# %%
# 誤差逆伝搬法部分
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# %%
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
# %%

# 確率最急降下法
optimizer = optim.SGD(net.parameters(), lr=0.01)

# トレーニング手順
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
