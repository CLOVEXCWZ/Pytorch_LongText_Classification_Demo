# -*- coding: utf-8 -*-
"""
FastText 基本做法：

    本网络只是模仿FastText，其过程并未完全按照FastText的思路实现网络。

    本网络主要是处理长文本的分类问题，其思路就是保留文本的句子级别和字/词级别，
 将一片文章处理成 n x m 的二维数据，其中n为句子数量，m为每个句子的词语数量。
 所以输入的数据便变成了 batch x n_sentence x n_words ，batch为每个训练批次大小，
 n_sentence为每篇文章保留句子数量，n_words为每一个句子中保留词语数量。

    将三维的数据进行embedding后对词语的维度进行最大池化，只保留词语中特征最显著的信息。
 然后再对句子的维度进行最大池化，取出保留句子中最显著的特征，
 最后接全连接层进行分类处理。

 说明，此种处理方式较为粗糙。

 网络的具体变换，直接查看 forward 方法，方法内附有详细的变换说明。

"""


import torch.nn as nn
import torch.nn.functional as F
import torch


class FastText(nn.Module):
    def __init__(self,
                 vocab_size,  # 词典的大小(总共有多少个词语/字)
                 n_class,     # 分类的类型
                 embed_dim=128,  # embedding的维度
                  ):

        super(FastText, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.fc = nn.Linear(in_features=embed_dim,
                            out_features=n_class)

    def forward(self, x):  # 以输入x为[32, 50, 100] 为例（32为batch_size, 50为保留句子数量，100为每一句保留词数量）
        x = self.embedding(x)  # [32, 50, 100, 128] 其中128为embedding的size，即将每个词/字转化为一个128维的稠密向量

        # 倒数第2维为词的维度，即将每一句的所有词进行最大池化，一个句子中只保留最显著的信息
        x = F.max_pool2d(x, (x.shape[-2], 1))  # [32, 50, 1, 128]
        x = x.squeeze()  # [32, 50, 128]  将多余的维度去掉
        x = F.max_pool2d(x, (x.shape[-2], 1)) # [32, 1, 128]  句子维度进行最大池化，只保留每篇文章最显著的信息
        x = x.squeeze()  # [32, 128]  去除多余的维度
        x = self.fc(x)   # [32, 4]  连接全连接进行分类（其中4为分类的类别数量）
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':

    vocab_size = 10000
    sentence_len = 50
    word_len = 100
    n_class = 4
    x_ = torch.ones((32, sentence_len, word_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = FastText(vocab_size, n_class)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    x_ = x_.to(device)
    y_ = y_.to(device)

    outputs = model(x_)
    optim.zero_grad()
    loss_value = loss(outputs, y_)
    loss_value.backward()
    optim.step()
    print(outputs.shape)











