# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import pickle as pkl
import jieba
from tqdm import tqdm
from public.path import *

# sougou_train_path = test_data_path
# sougou_dev_path = test_data_path
# segment_sougou_train_path = segment_test_data_path
# segment_sougou_dev_path = segment_test_data_path

np.random.seed(1)

__all__ = ['get_vocab', 'get_classs', 'load_dataset', 'dataIter']


def segment_texts(check_save=True):
    """ 检查并文本是否进行分词处理，只在word级别中才会用到
    :param check_save: 是否检查已保存文件, True已保存的文件不再进行处理，False忽略已经保存的文件
    """

    def jieba_segment(texts):
        """ 结巴分词处理，将所有文本进行分词处理，分词后的词语用空格连接 """
        for i, line in enumerate(tqdm(texts, "正在进行分词处理")):
            texts[i] = " ".join(jieba.cut(line))
        return texts

    def check_files(file_path, save_path, check_save):
        """ 检查文件，是否进行分词处理，若需要分词则进行分词并保存文件
        :param file_path: 需要进行分词的原文件地址
        :param save_path: 分词后保存的文件地址
        :param check_save: 是否检查已保存文件
        """
        if check_save and os.path.exists(save_path):
            pass
        else:
            if not os.path.exists(file_path):
                raise ValueError("搜狗语料训练集不存在，请检查")
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = f.read().split("\n")
                texts = [item.strip() for item in texts if len(item)>0]
            texts = jieba_segment(texts)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(texts))
    # 分别检查训练文本文件和验证文本文件
    check_files(sougou_train_path, segment_sougou_train_path, check_save)
    check_files(sougou_dev_path, segment_sougou_dev_path, check_save)


def get_vocab(leve='char', max_words=50000, min_freq=1, check_save=True):
    """ 获取词典，支持获取字符级别和词语级别，获取词语级别的时候，注意需要进行分词处理。
    :param leve:  字符级别或词语级别
    :param max_words:  词典保留最大词语数量
    :param min_freq:  最小出现的词频
    :param check_save: 是否需要检查以保存的文件
    :return: 词典的 char/word to index 字典
    """
    if leve not in ['char', 'word']:
        raise ValueError("model 只能是 char 或 word，请检查")
    if leve is 'char':
        path = sougou_train_path  # 去读词典文件
        save_path = char_vocab_path  # 词典文件
    else:
        segment_texts(check_save=True)  # 如果是词语级别的需要先检查是否进行分词
        path = segment_sougou_train_path
        save_path = word_vocab_path
    if check_save and os.path.exists(save_path):  # 检查已保存好的文件
        c2i = pkl.load(open(save_path, 'rb'))
        return c2i
    with open(path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [item.strip() for item in texts if len(item) > 0]
    word_dict = {}
    for line in texts:
        line_s = line.split('\t')
        if len(line_s) < 2:
            print(line_s)
            continue
        context, _ = line_s[0], line_s[1]  # 由于训练集存储格式为 data_text \t lable
        context = context.replace("\ue40c", " ") 
        if leve is 'word':
            context = context.split(" ")
        for word in context:
            if len(word) <= 0:
                continue
            word_dict[word] = word_dict.get(word, 0) + 1 
    word_sort = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    word_sort = [item for item in word_sort if item[1] >= min_freq]
    word_sort = word_sort[:max_words - 2] 
    words = ['[PAD]', '[UNK]'] + [item[0] for item in word_sort] 
    c2i = {c: i for i, c in enumerate(words)}
    pkl.dump(c2i, open(save_path, 'wb'))
    return c2i


def get_classs():
    """ 获取各个类型 """
    if not os.path.exists(sougou_class_path):
        raise ValueError("类别文本不存在，请检查！")
    with open(sougou_class_path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [str(item).strip() for item in texts if len(str(item).strip()) > 0]
    c2i = {class_: i for i, class_ in enumerate(texts)}
    return c2i


def load_dataset(model='train', leve='char', max_setence=10, max_words=50):
    """
    加载数据集，加载训练或验证数据集的data和label，并将文本转化为index形式
    :param model:  获取数据集的模式，'train'或者'dev'
    :param max_len: 每个样例保持的最长长度
    :return: samples
    """
    if leve not in ['char', 'word']:
        raise ValueError("model 只能是 char 或 word，请检查")
    if model not in ['train', 'dev']:
        raise ValueError("model 只能是 train、dev其中的一种，请检查")

    if model is 'train':
        if leve is 'char':
            file_path = sougou_train_path
        else:
            file_path = segment_sougou_train_path
    else:
        if leve is 'char':
            file_path = sougou_dev_path
        else:
            file_path = segment_sougou_dev_path
    if not os.path.exists(file_path):
        raise ValueError("文本不存在，请检查")
    print("正在打开文件,请稍后")
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = f.read().split("\n")
        texts = [item.strip() for item in texts if len(item) > 0]
    c2i = get_vocab(leve, max_words=50000, min_freq=1, check_save=True)
    pad_id = c2i.get('[PAD]', 0)
    samples_b = []
    for samples in tqdm(texts, desc="字符转index"):
        line_s = samples.split('\t')
        if len(line_s) < 2:
            continue
        context, label = line_s[0], line_s[1]

        lines_b = []
        for line in context.split("\ue40c"):
            if leve is 'word':
                line = line.split(" ")
            line_data = ([c2i.get(c, 1) for c in line if len(c) > 0])
            line_data = line_data + [pad_id] * (max_words - len(line_data))
            line_data = line_data[:max_words]
            lines_b.append(line_data)
        lines_b = lines_b[:max_setence]
        lines_b = lines_b + [[pad_id] * max_words] * (max_setence - len(lines_b))

        samples_b.append((lines_b, int(label.strip())))
    samples_b = np.array(samples_b)
    return samples_b


def shuffle_samples(samples):
    """ 打乱数据集 """
    samples = np.array(samples)
    shffle_index = np.arange(len(samples))
    np.random.shuffle(shffle_index)
    samples = samples[shffle_index]
    return samples


def dataIter(samples, batch_size=32, shuffle=True):
    """
    简易数据迭代器
    :param x: 数据数组
    :param y: 标签数组
    :param batch_size: 批次大小
    :param shuffle: 是否打乱数据
    :return: 返回一个数据迭代器
    """
    if shuffle:
        samples = shuffle_samples(samples)

    total = len(samples)
    n_bactch = total//batch_size
    for i in range(n_bactch):
        sub_samples = samples[i*batch_size: (i+1)*batch_size]
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        yield np.array(b_x), np.array(b_y)
    if total%batch_size != 0:  # 处理不够一个批次的数据
        sub_samples = samples[n_bactch * batch_size: total]
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        yield np.array(b_x), np.array(b_y)


class DataIter(object):
    """ 数据迭代器
    说明：Iter 主要需要实现 __next__ 、__iter__、__len__ 三个函数
    """
    def __init__(self, samples, batch_size=32, shuffle=True):
        if shuffle:
            samples = shuffle_samples(samples)
        self.samples = samples
        self.batch_size = batch_size
        self.n_batches = len(samples) // self.batch_size
        self.residue = (len(samples) % self.n_batches != 0)  # 是否为整数
        self.index = 0

    def split_samples(self, sub_samples):
        """ 用于分离data、lable等数据 """
        b_x = [item[0] for item in sub_samples]
        b_y = [item[1] for item in sub_samples]
        return np.array(b_x), np.array(b_y)

    def __next__(self):
        if (self.index == self.n_batches) and (self.residue is True):
            sub_samples = self.samples[self.index*self.batch_size: len(self.samples)]
            self.index += 1
            return self.split_samples(sub_samples)
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            sub_samples = self.samples[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self.split_samples(sub_samples)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


if __name__ == '__main__':
    # train_char = load_dataset(model='train', leve='char')
    # dev_char = load_dataset(model='dev', leve='char')
    # train_data = load_dataset()
    # train_iter = DataIter(train_data)
    # for x, y in train_iter:
    #     print(x.shape)

    w2i = get_vocab(leve='word', max_words=50000, min_freq=1, check_save=True)
    print(len(w2i))




