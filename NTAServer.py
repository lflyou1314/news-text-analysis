#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import jieba
import jieba.analyse
import flask
import configparser
import os
from snownlp import SnowNLP
from flask import jsonify
from flask import request
from flasgger import Swagger
import pickle
import fastText

# Create the application.
NTA = flask.Flask(__name__)
NTA_ROOT = os.path.dirname(os.path.abspath(__file__))
# Flasgger
Swagger(NTA)
# ConfigParser初始化
cf = configparser.ConfigParser()
cf.read(os.path.join(NTA_ROOT, "dev.cfg"), encoding='UTF-8')


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


@NTA.errorhandler(403)
def error_403(error):
    data = dict()
    data['ret'] = 403
    data['data'] = {}
    data['msg'] = 'error_403'
    return jsonify(data)


@NTA.errorhandler(404)
def error_404(error):
    data = dict()
    data['ret'] = 404
    data['data'] = {}
    data['msg'] = 'error_404'
    return jsonify(data)


@NTA.errorhandler(400)
def error_400(error):
    data = dict()
    data['ret'] = 400
    data['data'] = {}
    data['msg'] = 'error_400'
    return jsonify(data)


@NTA.errorhandler(410)
def error_410(error):
    data = dict()
    data['ret'] = 410
    data['data'] = {}
    data['msg'] = 'error_410'
    return jsonify(data)


@NTA.errorhandler(500)
def error_500(error):
    data = dict()
    data['ret'] = 500
    data['data'] = {}
    data['msg'] = 'error_500'
    return jsonify(data)


@NTA.route('/', methods=['GET'])
def index():
    """
            Web Service Status
            ---
            tags:
              - Awesome NTA API
            responses:
              500:
                description: Error The NTA is not awesome!
              200:
                description: ok
                schema:
                  id: NTA
                  properties:
                    ret:
                      type: integer
                      description: the status code
                    data:
                      type: string
                      description: the response data
                      default: ""
                    msg:
                      type: string
                      description: the response msg
                      default: ""
    """
    data = dict()
    data['ret'] = 200
    data['data'] = 'Serving Flask app - NTAServer'
    data['msg'] = 'success'
    return jsonify(data)


@NTA.route('/hello/<name>/<words>', methods=['GET'])
def hello(name, words):
    """
        Simple Example
        ---
        tags:
          - Awesome NTA API
        parameters:
          - name: name
            in: path
            type: string
            required: true
            description: the name of the person you want to tell
          - name: words
            in: path
            type: string
            required: true
            description: any words you want to say
        responses:
          500:
            description: Error The NTA is not awesome!
          200:
            description: ok
            schema:
              id: NTA
              properties:
                ret:
                  type: integer
                  description: the status code
                data:
                  type: array
                  description: the response data
                  items:
                    type: string
                  default: []
                msg:
                  type: string
                  description: the response msg
                  default: ""
    """
    data = dict()
    data['ret'] = 200
    data['data'] = {'name': name, 'words': words}
    data['msg'] = 'success'
    return jsonify(data)


# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch


# 读取文件
def readfile(filepath):
    with open(filepath, "r", encoding='UTF-8') as fp:
        content = fp.read()
    return content


@NTA.route('/analysis', methods=['POST'])
def analysis():
    """
        Get the analysis result
        ---
        tags:
          - Awesome NTA API
        parameters:
          - name: headline
            in: query
            type: string
            required: true
            description: the headline of the news
          - name: datacontent
            in: query
            type: string
            required: true
            description: the datacontent of the news
        responses:
          500:
            description: Error The NTA is not awesome!
          200:
            description: ok
            schema:
              id: NTA
              properties:
                ret:
                  type: integer
                  description: the status code
                data:
                  type: array
                  description: the response data
                  items:
                    type: string
                  default: []
                msg:
                  type: string
                  description: the response msg
                  default: ""
    """
    data = dict()
    headline = request.values.get('headline', None)
    datacontent = request.values.get('datacontent', None)
    if request.method == 'POST' and headline and datacontent:

        newstext = headline + datacontent

        # 基于 TextRank 算法的关键词抽取
        keywords_tfidf = jieba.analyse.extract_tags(newstext, topK=int(cf.get("jieba", "topK")))
        # 基于 TextRank 算法的关键词抽取
        keywords_textrank = jieba.analyse.textrank(newstext, topK=int(cf.get("jieba", "topK")))

        # 情感分析
        snownlp_handle = SnowNLP(newstext)
        sentiments = "{0:0.3}".format(snownlp_handle.sentiments)

        # 自动摘要
        summarys = snownlp_handle.summary()

        # 文本分类
        #############################################################################
        seg_list = get_seg_list(newstext)
        economy_lable = economy_classifier.predict(" ".join(seg_list))
        economy_lable_option = str(economy_lable[0][0]).split(",")[0][9:]
        economy_lable_probability = "{:.3f}".format(economy_lable[1][0])
        # print(cf.get("EconomyCategory", economy_lable_option))
        #############################################################################

        #############################################################################
        seg_list = get_seg_list(newstext)
        news_lable = news_classifier.predict(" ".join(seg_list))
        news_lable_option = str(news_lable[0][0]).split(",")[0][9:]
        news_lable_probability = "{:.3f}".format(news_lable[1][0])
        # print(cf.get("EconomyCategory", news_lable_option))
        #############################################################################


        data['ret'] = 200
        data['data'] = {'keywords_tfidf': keywords_tfidf, 'keywords_textrank': keywords_textrank,
                        'sentiments': sentiments, 'summarys': summarys,
                        'predicted_newscategory': cf.get("NewsCategory", news_lable_option),
                        'predicted_newscategory_code': news_lable_option,
                        'predicted_news_probability': news_lable_probability,
                        'predicted_economycategory': cf.get("EconomyCategory", economy_lable_option),
                        'predicted_economycategory_code': economy_lable_option,
                        'predicted_economy_probability': economy_lable_probability,
                        }
        data['msg'] = 'success'
    else:
        data['ret'] = 1001
        data['data'] = {}
        data['msg'] = 'parameter error'
    return jsonify(data)


if __name__ == '__main__':
    stopwords = get_stopwords(os.path.join(NTA_ROOT, "stop_words.txt"))

    NTA.debug = True
    # print(cf.get("Flask", "port"))

    # jieba 采用延迟加载,手动初始化
    jieba.initialize()

    economy_classifier = fastText.load_model(os.path.join(NTA_ROOT, "data/economycategory/economy.model.bin"))

    news_classifier = fastText.load_model(os.path.join(NTA_ROOT, "data/newscategory/news.model.bin"))

    NTA.run(host=cf.get("Flask", "host"), port=cf.get("Flask", "port"))
