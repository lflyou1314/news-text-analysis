#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import configparser
import os
import random
from pymongo import MongoClient
from datetime import datetime
import multiprocessing
import logging
import jieba
import sys
import fastText
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def waste_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        print("开始生成时间：{0}".format(datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S")))
        func(*args, **kwargs)
        end_time = datetime.now()
        print("结束生成时间：{0}".format(datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S")))
        m, s = divmod((end_time - start_time).seconds, 60)
        h, m = divmod(m, 60)
        print("总耗时：%02d:%02d:%02d" % (h, m, s))

    return wrapper


def get_stopwords(file_path):
    stopwords = []
    with open(file_path, "r", encoding='UTF-8') as fp:
        contents = fp.readlines()
        for content in contents:
            stopwords.append(content.strip())
    return stopwords


def get_seg_list(text):
    seg_list = jieba.cut(text)
    seg_list = filter(lambda x: len(x) > 1, seg_list)
    seg_list = filter(lambda x: x not in stopwords, seg_list)
    return seg_list


def writefile(sentences, fileName):
    logger.info("{0} writing data to fasttext format...".format(fileName))
    out = open(fileName, 'wb')
    for sentence in sentences:
        out.write(bytes(sentence, encoding="utf8").strip() + b"\n")
    logger.info("{} --- done!".format(fileName))


def init_config(configPath):
    cf = configparser.ConfigParser()
    cf.read(configPath, encoding='UTF-8')

    conn = MongoClient(cf.get("mongodb", "host"), int(cf.get("mongodb", "port")))
    db = conn.get_database("newsroom_unihub")
    newshub = db.manuscript.type.COMPO
    return cf, newshub


def progress_bar(num, total, option):
    rate = float(num) / total
    ratenum = int(100 * rate)
    sys.stdout.write("\r[{}{} ]\t{}%".format(' Loading ... EconomyCategory---> ', option, ratenum))
    sys.stdout.flush()


def load_task(option):
    query["category"] = option
    sub_sentences = []
    news_texts = newshub.find(query, projection);
    total = newshub.find(query, projection).count()
    count = 0
    for news in news_texts:
        # logger.info(news["id"])
        seg_list = get_seg_list(news["content"])
        sub_sentences.append("__lable__" + str(option) + "," + " ".join(seg_list))
        count += 1
        progress_bar(count, total, option)

    return sub_sentences


@waste_time
def load_data(cf, newshub):
    savepath = os.getcwd() + "\\data\\EconomyCategory\\economy.txt"

    try:
        for option in cf.options("EconomyCategory"):
            # logger.info("正在获取分类类别---{}".format(option))

            sentences.extend(load_task(option))
            sys.stdout.write("\t\t{} load completely \n".format(option))
            sys.stdout.flush()

    except Exception as e:
        logger.error(e.__str__())

    random.shuffle(sentences)
    writefile(sentences, savepath)


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)  # 创建大小为4的进程池
    sentences = []

    query = dict()
    projection = dict()
    projection["id"] = 1
    projection["content"] = 1
    projection["publishDate"] = 1
    projection["properties.fileName"] = 1

    stopwordsFilePath = r"./stop_words.txt"
    stopwords = get_stopwords(stopwordsFilePath)

    configPath = r"./dev.cfg"
    cf, newshub = init_config(configPath)
    # print(cf.get("mongodb", "host"))

    query["folder"] = cf.get("mongodb", "zscpg")

    # load_data(cf, newshub)

    classifier = fastText.train_supervised(r"./data/economycategory/economy.txt", lr=0.6, label="__lable__", wordNgrams=2,
                                           thread=24, epoch=25, verbose=2)

    classifier.save_model(r"./data/economycategory/economy.model")
    # texts = ['新华财经沈阳8月29日电（记者李宇佳）全国工商联副主席、TCL集团创始人、董事长李东生29日在沈阳举行的2018中国民营企业500强峰会上表示，掌握核心技术是企业立身之本，注入新动能才能助推企业高质量发展。李东生表示，目前我国民营经济进入到发展的最好时期，发展质量和效益稳步提升，已成为创新的重要动力和来源，对外投资保持强劲势头。但随着国际市场竞争加剧，民营企业的发展仍面临许多考验和不确定性。民营企业需要寻找和转换新的发展动能，形成更强的核心竞争力，才能适应新的竞争环境，实现更高质量的发展。他认为，民营企业应该走向产业链的高端，拓展新的价值空间，同时还要掌握核心技术，因为这是一家企业的立身之本。此外，要优化海外产业布局，创新商业模式，激发企业增长新动力。李东生表示，未来TCL将聚焦三个最前沿的关键技术领域进行布局：人工智能、互联网应用和大数据；半导体印刷显示技术与材料；智能制造和工业互联网。努力将TCL从工业品制造企业发展为一个产品＋服务、智能＋互联网的新型企业。（编辑：杜少军）']
    # seg_list = get_seg_list(texts[0])
    # lable = classifier.predict(" ".join(seg_list))
    # lable_option = str(lable[0][0]).split(",")[0][9:]
    # print(lable_option)
    # print(cf.get("EconomyCategory",lable_option))
    pool.close()
    pool.join()
