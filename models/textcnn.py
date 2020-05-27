# -*- coding: utf-8 -*-
"""
TextCNN 基本做法：

    本网络只是模仿TextCNN，其过程并未完全按照FastText的思路实现网络。

    本网络主要是处理长文本的分类问题，其思路就是保留文本的句子级别和字/词级别，
 将一片文章处理成 n x m 的二维数据，其中n为句子数量，m为每个句子的词语数量。
 所以输入的数据便变成了 batch x n_sentence x n_words ，batch为每个训练批次大小，
 n_sentence为每篇文章保留句子数量，n_words为每一个句子中保留词语数量。

    将三维的数据进行embedding后对词语的维度进行最大池化，只保留词语中特征最显著的信息。
 然后再进行2、3、4的长度卷积操作，类似于2、3、4 gram，取出保留句子中最显著的特征，
 最后接全连接层进行分类处理。

 说明：TextCNN 卷积输出特征为2个通道，这边采用更多的输出通道。

 网络的具体变换，直接查看 forward 方法，方法内附有详细的变换说明。
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self,
                 vocab_size,  # 词典的大小(总共有多少个词语/字)
                 n_class,  # 分类的类型
                 embed_dim=300,  # embedding的维度
                 num_filters=256,  # 等于2的效果会比较差，等于256的效果会比较好
                 ):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (f, embed_dim)) for f in [2, 3, 4]])
        self.fc = nn.Linear(in_features=num_filters * 3,
                            out_features=n_class)

    def forward(self, x):  # 以输入x为[32, 50, 100] 为例（32为batch_size, 50为保留句子数量，100为每一句保留词数量）
        x = self.embedding(x)  # [32, 50, 100, 300] 其中128为embedding的size，即将每个词/字转化为一个128维的稠密向量

        # 倒数第2维为词的维度，即将每一句的所有词进行最大池化，一个句子中只保留最显著的信息
        x = F.max_pool2d(x, (x.shape[-2], 1))  # [32, 50, 1, 300]
        x = x.squeeze()  # [32, 50, 300]  将多余的维度去掉
        x = x.unsqueeze(1)   # [batch, 1, seq_len, embed_dim]   增加通道维度
        print(x.shape)
        pooled = []
        for conv in self.convs:
            out = conv(x)   # [32, 256, 49(或者48或47), 1] 经过卷积后的大小，256为卷积输出通道
            out = F.relu(out)
            out = F.max_pool2d(out, (out.shape[-2], 1)) # [32, 256, 1, 1] 进行池化操作
            out = out.squeeze()  # [32, 256] 去掉多余的维度
            pooled.append(out)
        x = torch.cat(pooled, dim=-1)  # [32, 768] 将三次卷积获得的数据进行拼接操作
        x = self.fc(x)  # 接全连接进行分类
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':

    vocab_size = 10000
    sentence_len = 50
    word_len = 100
    n_class = 4
    x_ = torch.ones((32, sentence_len, word_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = TextCNN(vocab_size, n_class)
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










