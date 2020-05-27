# -*- coding: utf-8 -*-
# 定义一些默认文件、文件夹的路径

import os

cur_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件夹地址

data_set_dir = os.path.join(cur_dir, "../datas")          # 数据集地址
sougou_news_dir = os.path.join(data_set_dir, "SougouNews")      # 搜狗数据集地址
sougou_train_path = os.path.join(sougou_news_dir, "sougou_train.txt")  # 搜狗训练文件地址
sougou_dev_path = os.path.join(sougou_news_dir, "sougou_dev.txt")  # 搜狗验证集文件地址
sougou_class_path = os.path.join(sougou_news_dir, "class.txt")  # 数据集类型地址

test_data_path = os.path.join(sougou_news_dir, "test_data.txt")  # 测试数据集（调试代码时使用的小数据集）
segment_test_data_path = os.path.join(sougou_news_dir, "segment_test_data.txt")  # 分词后侧测试数据集（调试代码时使用）

char_vocab_path = os.path.join(sougou_news_dir, "char_vocab.pkl")  # 字级别词典
word_vocab_path = os.path.join(sougou_news_dir, "word_vocab.pkl")  # 词级别词典

segment_sougou_train_path = os.path.join(sougou_news_dir, "segment_train.txt")  # 分词后的训练集
segment_sougou_dev_path = os.path.join(sougou_news_dir, "segment_dev.txt")  # 分词后的验证集

log_dir = os.path.join(cur_dir, "log")  # 日志地址

# 目录初始化
def init_dir():
    """ 初始化文件夹地址，将未创建的文件夹进行创建操作 """
    dirs = [data_set_dir, log_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)










