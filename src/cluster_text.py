#!/usr/bin/env python
# coding=utf-8

import os,sys
import jieba
import codecs
import numpy as np
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from zh_wiki import *
from langconv import *

from collections import defaultdict

import logging
import simplejson as json

reload(sys)
sys.setdefaultencoding("utf-8")

logging.basicConfig( format = "%(asctime)s: %(levelname)s : %(message)s:", level = logging.INFO, filename = "logfile.log", filemode = "w")


class TextCluster(object):
    """
    short text cluster using tfidf and kmeans
    """

    def __init__(self):
        """
        initialize stop_words and langconvert handler
        """

        self.STOP_WORDS = set([line.strip() for line in codecs.open("./data/stop_words.txt", "rb", "utf-8")])
        self.langConvHandler = Converter("zh-hans") # convert Chinese from complex to simple

    
    def gen_data(self, fname):
        """
        :fname : input file, every line means a single data
        :rtype : List[List[float]]: data matrix
        """
        
        lines = [ self.langConvHandler.convert(line.strip().lower()) for line in codecs.open(fname, "rb","utf-8") if len(line) > 6]
        # lines = list(set(lines))  # remove duplicates
        
        
        logging.info("number of data %d " % len(lines))
        cut_lines = [" ".join(jieba.cut(line)) for line in lines]

        # transform to tfidfVec
        tfidfVec = TfidfVectorizer(max_features = 3000)
        tfidf_data = tfidfVec.fit_transform(cut_lines)
        tfidf_data = tfidf_data.toarray()
       
        # save origin text
        with open("./output/origin_lines.txt", "wb") as fw:
            json.dump(lines, fw)
        
        # save vectorize data
        np.save("./output/tfidf.corpus.npy", tfidf_data)
        
        self.lines = lines
        self.tfidf_data = tfidf_data



    def gen_cluster(self, k = -1, trained = False):
        """
        gen cluster with kmeans
        """

        if trained:
            lines = json.load(open("./output/origin_lines.txt"))
            data = np.load("./output/tfidf.corpus.npy")
        else:
            lines, data = self.lines, self.tfidf_data

        if k == -1:
            k = int(2*np.sqrt(len(lines))) if len(lines)<100 else int(3*np.sqrt(len(lines)))
        elif k>1:
            k = int(k)

        km = cluster.KMeans(n_clusters = k, precompute_distances = True, max_iter = 500, n_jobs = -1).fit(data)
        # db = cluster.DBSCAN(eps = 3.0, algorithm = "brute", metrix = "cosine", min_samples = 5).fit(data)

        logging.info("km cluster arguments %s " % km )

        labels = km.labels_
    #    centers = km.cluster_centers_
        logging.info("cluster result labels")
        logging.info(labels[:100])

        res = defaultdict(list)
        for label, line in zip(labels,lines):
            res[label].append(line)

        sorted_res = sorted(res.iteritems(), key = lambda x: -len(x[1]))
        
        result = []
        for label, lines in sorted_res:
            tmp = {}
            tmp["cluster"] = str(label+1)
            tmp["lines"] = lines
            result.append(tmp)

        with open("./output/cluster_result.txt", "wb") as fw:
            json.dump(result, fw, ensure_ascii = False, indent = 4)

        return result


if __name__ == "__main__":

    test = TextCluster()
    test.gen_data(fname="./data/cleaned_data.txt")
    test.gen_cluster()





