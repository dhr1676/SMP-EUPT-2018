import pandas as pd
import numpy as np
import codecs
import jieba
import jieba.posseg
import time

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from joblib import Parallel, delayed
import logging


def del_stop_words(words, stop_words_set):
    if type(words) != type(" "):
        return ""
    result = jieba.posseg.cut(words)
    new_words = []
    for word in result:
        if word.word not in stop_words_set and word.flag is not "x":
            new_words.append(word.word)
    ans = ' '.join(new_words)
    return ans if len(ans) > 0 else ""


def get_label(label):
    if label == "自动摘要":
        return 0

    if label == "机器翻译":
        return 1

    if label == "机器作者":
        return 2

    if label == "人类作者":
        return 3


def restore_label(number):
    if number == 0:
        return "自动摘要"

    if number == 1:
        return "机器翻译"

    if number == 2:
        return "机器作者"

    if number == 3:
        return "人类作者"


"""
自动摘要0: 31031
机器翻译1: 36180
机器作者2: 31153
人类作者3: 47977
"""

if __name__ == '__main__':
    start_time = time.time()
    # ----------------Set Path----------------------------------------
    train_path = "./input/training_new_split.csv"
    test_path = "./input/validation_new_split.csv"

    # train_path = "./input/trim_2k_split.csv"
    # test_path = "./input/test_2k_split.csv"
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    # ----------------------------------------------------------------

    # ----------------Read Train Data---------------------------------
    raw_train = pd.read_csv(train_path, encoding="utf-8")
    print("Training data shape", raw_train.shape)
    # print(raw_train.describe())
    # ----------------------------------------------------------------

    # ----------------Read Test Data----------------------------------
    test_data = pd.read_csv(test_path, encoding="utf-8", usecols=[1])
    print("Test data shape", test_data.shape)
    # print(test_data.describe())
    # ----------------------------------------------------------------

    # # ----------------Get Label-----------------------------
    # raw_train["Label"] = Parallel(n_jobs=4, verbose=10)(
    #     delayed(get_label)(raw_train.iloc[index]["Label"]) for index in range(len(raw_train)))
    # # ----------------------------------------------------------------

    # ----------------Remove Stop Words-----------------------------
    stop_words_path = "./input/stop_words_zh.txt"
    stop_list = [line.strip() for line in codecs.open(stop_words_path, 'r', encoding='gb2312').readlines()]
    stop_words = set(stop_list)
    if len(stop_words) > 0:
        print("Got stop words set!\n")

    raw_train["Content"] = Parallel(n_jobs=4, verbose=10)(
        delayed(del_stop_words)(raw_train.iloc[index]["Content"], stop_words) for index in range(len(raw_train)))

    raw_train["Label"] = Parallel(n_jobs=4, verbose=10)(
        delayed(get_label)(raw_train.iloc[index]["Label"]) for index in range(len(raw_train)))

    raw_train = raw_train.dropna(subset=["Content"])
    print("Training data shape after remove stop words\n", raw_train.shape)

    test_data["Content"] = Parallel(n_jobs=4, verbose=10)(
        delayed(del_stop_words)(test_data.iloc[index]["Content"], stop_words) for index in range(len(test_data)))

    # test_data["Label"] = Parallel(n_jobs=4, verbose=10)(
    #     delayed(get_label)(test_data.iloc[index]["Label"]) for index in range(len(test_data)))

    test_data = test_data.dropna(subset=["Content"])
    print("Test data shape after remove stop words\n", test_data.shape)

    # ----------------------------------------------------------------

    x_train = raw_train["Content"].values
    y_train = raw_train["Label"].values
    del raw_train

    x_test = test_data["Content"].values
    # y_test = test_data["Label"].values
    del test_data

    # ------------------Data set-----------------------------
    print("Start Vectorization...")
    all_text = np.append(x_train, x_test)
    count_v0 = CountVectorizer()
    counts_all = count_v0.fit_transform(all_text)

    count_v1 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_train = count_v1.fit_transform(x_train)
    print("the shape of train is " + repr(counts_train.shape))

    count_v2 = CountVectorizer(vocabulary=count_v0.vocabulary_)
    counts_test = count_v2.fit_transform(x_test)
    print("the shape of test is " + repr(counts_test.shape))

    tfidftransformer = TfidfTransformer()
    train_data = tfidftransformer.fit(counts_train).transform(counts_train)
    test_data = tfidftransformer.fit(counts_test).transform(counts_test)
    # ----------------------------------------------------------------

    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, y_train)
    predicts = clf.predict(test_data)
    predicts = predicts.tolist()

    submission = pd.DataFrame(predicts, columns=["Predict"])
    print("Submission shape\n", submission.shape)
    # submission.to_csv("./output/0704_sub_NB_all.csv", index=False)

    end_time = time.time()
    print("Running time: %f s\n" % (end_time - start_time))

    index = pd.read_csv("./input/validation_new.csv", encoding="utf-8", usecols=[0])
    submission = pd.read_csv("./output/0704_sub_NB_all.csv", encoding="utf-8")
    submission["Predict"] = Parallel(n_jobs=4, verbose=10)(
        delayed(restore_label)(submission.iloc[index]["Predict"]) for index in range(len(submission)))

    submission = pd.concat([index, submission], axis=1)
    submission.to_csv("./output/0704_result_NB_all.csv", index=False, header=None)
