# import time
# import logging
# import multiprocessing
#
# from joblib import Parallel, delayed
#
# import pandas as pd
import fasttext


# def get_label(line):
#     if line["Label"] == "自动摘要":
#         return "__label__0" + "," + line["Content"]
#
#     if line["Label"] == "机器翻译":
#         return "__label__1" + " , " + line["Content"]
#
#     if line["Label"] == "机器作者":
#         return "__label__2" + "," + line["Content"]
#
#     if line["Label"] == "人类作者":
#         return "__label__3" + "," + line["Content"]


if __name__ == '__main__':
    # # ----------------Set Path----------------------------------------
    # start_time = time.time()
    # NUM_CORES = multiprocessing.cpu_count()
    # # train_path = "./input/training_new.csv"
    # train_path = "./input/trim_2k_split.csv"
    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger(__name__)
    # # ----------------------------------------------------------------
    #
    # # ----------------Read Train Data---------------------------------
    # raw_train = pd.read_csv(train_path, encoding="utf-8")
    # print("Training raw data shape", raw_train.shape)
    # raw_train = raw_train.dropna(subset=["Content"])
    # print("Training data drop nan shape", raw_train.shape)
    # # ----------------------------------------------------------------
    #
    # # ----------------Shuffle Train Data--------------------------------
    # # raw_train = shuffle(raw_train)
    # raw_train = raw_train.sample(frac=1).reset_index(drop=True)
    # print("Training data shape after shuffle", raw_train.shape)
    #
    # # ----------------Get Label-----------------------------
    # raw_train["Label"] = Parallel(n_jobs=NUM_CORES, verbose=10)(
    #     delayed(get_label)(raw_train.iloc[index]) for index in range(len(raw_train)))
    # del raw_train["Content"]
    # ----------------------------------------------------------------

    # raw_train.to_csv("./input/trim_2k_ft.csv", index=False, header=None)
    # raw_train = pd.read_csv("./input/trim_2k_ft.csv", encoding="utf-8")
    # raw_train.to_csv("./input/trim_2k_ft.txt", index=False, header=None)

    # model = fasttext.skipgram('./input/trim_2k_ft.txt', './model/word_model')
    model = fasttext.load_model("./model/word_model.bin")
    print(model.words)

    # classifier = fasttext.supervised("./input/trim_2k_ft.txt", './model/classifier.model', label_prefix='__label__')
    # result = classifier.test("./input/trim_2k_ft.txt")
    #
    # print("P@1:", result.precision)  # 准确率
    # print("R@2:", result.recall)  # 召回率
    # print("Number of examples:", result.nexamples)  # 预测错的例子
