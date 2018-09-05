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
    sys.stdout.write("\r[{}{} ]\t{}%".format(' Loading ... NewsCategory---> ', option, ratenum))
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
    savepath = os.getcwd() + "\\data\\newscategory\\news.txt"

    try:
        for option in cf.options("NewsCategory"):
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

    # classifier = fastText.train_supervised(r"./data/newscategory/news.txt", lr=0.6, label="__lable__", wordNgrams=2,
    #                                        thread=24, epoch=25, verbose=2)
    # classifier.save_model(r"./data/newscategory/news.model")
    classifier = fastText.load_model(r"./data/newscategory/news.model")

    print(datetime.now())
    texts = [
        '新华社悉尼８月２９日电（陈宇）澳大利亚昆士兰大学研究人员用基因测序技术绘制出细菌进化树，这一方法有助于改进以往的细菌分类法。 研究成果日前发表在英国《自然·生物技术》杂志上。研究人员主要使用一种被称为元基因组学的方法，对自然环境中的细菌样本直接进行测序，进而绘制出细菌进化树，更好地对细菌进行分类。 现有的生物分类主要依据生物在形态结构和生理功能等方面的特征，以弄清不同类群之间的亲缘关系和进化关系。 研究项目负责人、昆士兰大学化学和分子生物科学学院教授菲利普·胡金霍尔茨说，尽管科学界总体认可这种根据生物进化关系来确定的分类，但因为细菌难以根据物理特征区分，所以此前对细菌的分类存在一些错误。 研究人员以细菌中最常见的１２０种基因图谱为基础绘制出庞大的细菌进化树，以构建一个标准化的模型，修正以前的分类错误。 例如，按照原先的分类方法，只要是细胞内部产生孢子的杆状细菌都被归入梭菌属，而现在可以根据进化树将梭菌属下的细菌重新分类到２９个科的１２１个不同的属。 研究团队中负责软件开发的博士多诺万·帕克斯说，随着生物测序技术的进步，现在研究人员可以获得成千上万种细菌的完整基因“蓝图”，包括一些目前还无法在实验室培养出的细菌。（完）']
    seg_list = get_seg_list(texts[0])

    lable = classifier.predict(" ".join(seg_list))
    print(lable)
    print(datetime.now())
    print("1111111111111111")
    print(datetime.now())
    texts = [
        '新华社北京8月29日电外交部发言人华春莹29日宣布：中非合作论坛第七届部长级会议将于9月2日在北京举行。国务委员兼外交部长王毅、商务部部长钟山将同论坛共同主席国南非国际关系与合作部长和贸易与工业部长共同主持会议。论坛其他53个非方成员负责外交和对外经济合作事务的部长或代表将出席会议。此次会议主要任务是为9月3日至4日召开的2018年中非合作论坛北京峰会作准备。（完）']
    seg_list = get_seg_list(texts[0])

    lable = classifier.predict(" ".join(seg_list))
    print(lable)
    print(datetime.now())


    pool.close()
    pool.join()
