# -*- coding: utf-8 -*-
"""
搜狗新闻语料解析
语料下载地址：http://www.sogou.com/labs/resource/list_news.php

"""
import os
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd


def get_files_path(base_path):
    """ 读取 .txt 文件 """
    if not os.path.exists(base_path):
        raise Exception(f"文件地址不存在，请检查 {base_path}")
    if not os.path.isdir(base_path):
        raise Exception(f"文件地址不是文件夹，请检查 {base_path}")
    files = os.listdir(base_path)
    files = [item for item in files if ".txt" in item]
    path = [os.path.join(base_path, item) for item in files]
    return path


def read_files(files_path, save_path, url_map=dicurl, encoding="gb18030"):
    sampes = []
    totals = len(files_path)
    for i, file in enumerate(files_path):
        print(f"正在处理: {i + 1}/{totals}")
        try:
            with open(file, 'r', encoding=encoding) as f:
                text = f.read()
        except FileNotFoundError:
            print('无法打开指定的文件!')
        except LookupError:
            print('指定了未知的编码!')
        except UnicodeDecodeError:
            print('读取文件时解码错误!')
        except Exception:  # 其他异常
            print("其他异常")
        text_ = "<docs>\n" + text + "</docs>"
        soup = BeautifulSoup(text_, 'html.parser')
        i = 0
        for item in soup.docs.contents:
            if len(item) <= 1:
                continue
            url = item.url.string
            docno = item.docno.string
            contenttitle = item.contenttitle.string
            content = item.content.string

            url_sub = re.findall("http://[a-zA-Z0-9.]+", url)[0]
            url_sub = str(url_sub).replace("http://", "")
            news_type = str(url_sub).replace(".sohu.com", "")

            a_sample = [news_type, url, docno, contenttitle, content]
            sampes.append(a_sample)
            # 进行归类处理
    sampes = np.array(sampes)
    df = pd.DataFrame()
    for i, col in enumerate(['type', 'url', 'docno', 'contenttitle', 'content']):
        df[col] = sampes[:, i]
    df.to_csv(save_path, index=False)
    return df


def split_stypes(df, types=['sports', 'news', 'house', 'business']):
    """ 将一些类型分离出来 """
    selected = df.type.isin(types)
    return df[selected]

def text_clear(x):   
    
    x = str(x) 
    x = x.lower()    
    regexp_flux = re.compile('[\s+\!_$%^*(+\"\')]+|[+——()?【】。“”！？、~@#￥%……&*（）；：{}~……《》<>「」]+')
    x = re.sub(regexp_flux, '', x) 
    regexp_date = re.compile('[\s+\!_$%^*(+\"\')]+|[+——()?【】。“”！？、~@#￥%……&*（）；：{}~……《》<>「」]+')
    x = re.sub(regexp_date, '', x)    
    blank = re.compile(r'\s+')
    x = re.sub(blank, ' ', x) 
    
    return x 


if __name__ == '__main__':
    # 搜狗语料存储文件夹
    text_dir = "/Users/zhouwencheng/Downloads/SogouCS.reduced/"
    save_path = os.path.join(text_dir, "搜狗语料文件.csv")

    files_path = get_files_path(text_dir)
    df = read_files(files_path, save_path)
    """
    sports      85984
    news        82740
    house       71221
    business    61843
    yule        33091
    2008        27034
    women       16887
    it          12353
    learning    10673
    travel       8957
    auto         7241
    health       5482
    cul          3291
    mil.news     2930
    career         92 
    """

    # 分离出几个类型比较均衡的数据
    sub_save_path = os.path.join(text_dir, "搜狗新闻语料摘取四种样本均衡样本.csv")
    types = ['sports', 'news', 'house', 'business']
    sub_df = split_stypes(df, types)
    sub_df.to_csv(sub_save_path, index=False)
    """
    sports      85984
    news        82740
    house       71221
    business    61843 
    """




