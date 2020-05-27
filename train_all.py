# -*- coding: utf-8 -*-

import torch

from dataprocess import get_vocab, load_dataset, DataIter, get_classs
from models.fasttext import FastText
from models.textcnn import TextCNN
from models.textrcnn import TextRCNN
from models.textrnn import TextRNN
from models.transformer import Transformer

from train import train, create_log
from public.path import log_dir
import os
import time

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


def get_models(vocab_size,  # 词典大小
               n_class=10,  # 类别个数
               seq_len=38,  # 句子长度
               device=None):  # 设备
    """ 获取所有需要训练的模型 """

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fast_text = FastText(vocab_size=vocab_size, n_class=n_class)
    text_cnn = TextCNN(vocab_size=vocab_size, n_class=n_class)
    text_rnn = TextRNN(vocab_size=vocab_size, n_class=n_class)
    text_rcnn = TextRCNN(vocab_size=vocab_size, n_class=n_class)
    transformer = Transformer(vocab_size=vocab_size, seq_len=seq_len,
                              n_class=n_class, device=device)
    return [fast_text, text_cnn, text_rnn, text_rcnn, transformer]
    # return [transformer]


if __name__ == '__main__':

    leve = 'char'
    max_sentence = 10
    max_words = 50

    # c2i = get_vocab()   # 获取词典
    c2i = get_vocab(leve=leve, max_words=50000, min_freq=1, check_save=True)
    class_ = get_classs()  # 获取数据集的类别

    max_len_ = 10
    n_class_ = len(class_)  # 类别的数量
    vocab_size_ = len(c2i)  # 词典大小
    epochs = 100  # 训练周期
    stop_patience = 5  # 提前终止周期（当验证集损失函数连续 stop_patience 个周期没有减少小于最小值时终止训练）

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取数据集，同时创建迭代器
    # train_samples = load_dataset('train', max_len=max_len_)
    # dev_samples = load_dataset('dev', max_len=max_len_)

    train_samples = load_dataset(model='train', leve=leve, max_setence=max_sentence, max_words=max_words)
    dev_samples = load_dataset(model='dev', leve=leve, max_setence=max_sentence, max_words=max_words)

    train_iter = DataIter(train_samples)
    dev_iter = DataIter(dev_samples)

    models = get_models(vocab_size=vocab_size_,  # 词典大小
                        n_class=n_class_,  # 类别个数
                        seq_len=max_sentence,  # 句子长度
                        device=device)

    # 创建日志
    log_path = os.path.join(log_dir, f"{time.strftime('%Y-%m-%d_%H_%M')}.log")
    log = create_log(path=log_path)  # 获取日志文件

    log.info(f"训练集数量:{len(train_samples)} 测试集样本数量:{len(dev_samples)} ")
    log.info(f"embedding级别:{leve} 词典大小:{vocab_size_} 每个样本保留最长句子数:{max_sentence} 每句保留最长词\字数:{max_words}")

    # 训练所有的样本
    for model in models:
        train(model,  # 需要训练的模型
              train_iter,  # 训练数据迭代器）
              dev_iter=dev_iter,  # 验证数据迭代器
              epochs=epochs,  # 训练周期
              stop_patience=stop_patience,  # 提前终止周期
              device=device,
              log=log)  # 记录训练数据



