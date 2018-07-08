import logging
import time
from joblib import Parallel, delayed
import multiprocessing


import pandas as pd
import numpy as np
import jieba
import jieba.posseg

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Sequential

from sklearn.utils import shuffle


def del_stop_words_train(words, stop_words_set):
    if type(words) != type(" "):
        return np.nan
    result = jieba.posseg.cut(words)
    new_words = []
    for word in result:
        if word.word not in stop_words_set and word.flag is not "x":
            new_words.append(word.word)
    ans = ' '.join(new_words)
    return ans if len(ans) > 0 else np.nan


def get_label(label):
    if label == "自动摘要":
        return 0

    if label == "机器翻译":
        return 1

    if label == "机器作者":
        return 2

    if label == "人类作者":
        return 3


"""
自动摘要0: 31031
机器翻译1: 36180
机器作者2: 31153
人类作者3: 47977
"""

if __name__ == '__main__':
    # ----------------Set Path----------------------------------------
    start_time = time.time()
    NUM_CORES = multiprocessing.cpu_count()
    train_path = "./input/train_pseudo_label.csv"
    # train_path = "./input/training_new_split.csv"
    # train_path = "./input/trim_2k_split.csv"
    test_path = "./input/validation_new_split.csv"
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    # ----------------------------------------------------------------

    # ----------------Read Train Data---------------------------------
    raw_train = pd.read_csv(train_path, encoding="utf-8")
    print("Training raw data shape", raw_train.shape)
    raw_train = raw_train.dropna(subset=["Content"])
    print("Training data drop nan shape", raw_train.shape)
    # print(raw_train.describe())
    # ----------------------------------------------------------------

    # ----------------Read Test Data----------------------------------
    test_data = pd.read_csv(test_path, encoding="utf-8", usecols=[1])
    test_data = test_data.fillna(method="ffill")
    print("Test data shape", test_data.shape)
    # print(test_data.describe())
    # ----------------------------------------------------------------

    # ----------------Shuffle Train Data--------------------------------
    # raw_train = shuffle(raw_train)
    raw_train = raw_train.sample(frac=1).reset_index(drop=True)
    print("Training data shape after shuffle", raw_train.shape)

    # ----------------Get Label-----------------------------
    raw_train["Label"] = Parallel(n_jobs=NUM_CORES, verbose=10)(
        delayed(get_label)(raw_train.iloc[index]["Label"]) for index in range(len(raw_train)))
    # ----------------------------------------------------------------

    # ----------------Set Parameters-----------------------------
    print("Start Pad Sequence...\n")
    MAX_SEQUENCE_LENGTH = 500       # 每条新闻最大长度
    EMBEDDING_DIM = 200             # 词向量空间维度
    VALIDATION_SPLIT = 0.16         # 验证集比例
    TEST_SPLIT = 0                  # 测试集比例
    EPOCHS = 3                      # 迭代轮次

    labels = to_categorical(np.asarray(raw_train["Label"]))
    del raw_train["Label"]

    p1 = len(raw_train)

    tokenizer = Tokenizer()
    all_text = raw_train.append(test_data)

    tokenizer.fit_on_texts(all_text["Content"])
    sequences = tokenizer.texts_to_sequences(all_text["Content"])

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    x_train = data[:p1]
    y_train = labels[:p1]

    x_test = data[p1:]

    print('train docs: ' + str(len(x_train)))
    print('test docs: ' + str(len(x_test)))

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(Dropout(0.4))
    model.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dense(labels.shape[1], activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    print("Start training!..\n")
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, validation_split=VALIDATION_SPLIT)
    model.save('cnn_PL.h5')
    y_predict = model.predict(x_test)
    print(type(y_predict))
    print("y_predict type", type(y_predict))
    print("y_predict shape", y_predict.shape)

    # ----------------Output to the File------------------------------
    submission = pd.DataFrame(y_predict)

    submission.to_csv("./output/0705_sub_CNN.csv", index=False)

    logger.info("Training is Done!\n")

    end_time = time.time()
    print("Running time: %f s\n" % (end_time - start_time))
