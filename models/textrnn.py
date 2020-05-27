# -*- coding: utf-8 -*-
"""
TextRNN 基本做法：


    本网络只是模仿TextRNN，其过程并未完全按照TextRNN的思路实现网络。

    本网络主要是处理长文本的分类问题，其思路就是保留文本的句子级别和字/词级别，
 将一片文章处理成 n x m 的二维数据，其中n为句子数量，m为每个句子的词语数量。
 所以输入的数据便变成了 batch x n_sentence x n_words ，batch为每个训练批次大小，
 n_sentence为每篇文章保留句子数量，n_words为每一个句子中保留词语数量。

    将三维的数据进行embedding后对词语的维度进行最大池化，只保留词语中特征最显著的信息。
 然后再进行RNN操作，
 最后接全连接层进行分类处理。

 网络的具体变换，直接查看 forward 方法，方法内附有详细的变换说明。


"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRNN(nn.Module):

    def __init__(self,
                 vocab_size,  # 词典的大小(总共有多少个词语/字)
                 n_class,  # 分类的类型
                 embed_dim=300,  # embedding的维度
                 rnn_hidden=256,
                 ):
        super(TextRNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
        )
        self.gru = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_hidden,
                          batch_first=True)
        self.fc = nn.Linear(in_features=rnn_hidden,
                            out_features=n_class)

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, max_words, embed_dim]
        x = F.max_pool2d(x, (x.shape[-2], 1))  # [batch, max_sentence, 1, embed_dim]
        x = x.squeeze()  # [batch, max_sentence, embed_dim]

        # output:[batch, seq_len, rnn_hidden]
        # h_n:[1, batch, rnn_hidden]
        # c_n:[1, batch, rnn_hidden]
        output, h_n = self.gru(x)
        x = h_n.squeeze()  # [batch, rnn_hidden]
        x = self.fc(x)  # [batch, n_class]
        x = torch.sigmoid(x)
        return x


if __name__ == '__main__':

    vocab_size = 10000
    sentence_len = 50
    word_len = 100
    n_class = 4
    x_ = torch.ones((32, sentence_len, word_len)).to(torch.int64)
    y_ = torch.ones((32,)).to(torch.int64).random_(n_class)

    model = TextRNN(vocab_size, n_class)
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
