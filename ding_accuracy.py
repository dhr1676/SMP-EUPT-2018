import pandas as pd
import re

if __name__ == '__main__':
    # ----------------Read Data----------------
    data_path = "./output/submission_predict_log_old.csv"
    sample_path = "./input/validation.csv"
    ss_path = "./output/eupt_sample_submission.csv"
    raw_data = pd.read_csv(data_path, encoding="utf-8")
    sample_data = pd.read_csv(sample_path, encoding="utf-8", header=0, usecols=[0])
    ss_data = pd.read_csv(ss_path, encoding="utf-8", header=0, usecols=[0])

    # print(sample_data.columns.tolist())
    # print(sample_data.head(5))
    print(len(sample_data))
    print(len(raw_data))
    print(len(ss_data))

    columns = raw_data.columns.tolist()

    lst = []
    for i in range(len(raw_data)):
        lst.append(raw_data.iloc[i].values.tolist())

    index = []

    for i in range(len(lst)):
        # print(type(lst), type(lst[i]))
        index.append(lst[i].index(max(lst[i])))
        # print(i, index[i], max(lst[i]))

    predict = []
    for i in range(len(index)):
        predict.append(columns[index[i]])

    print(len(predict))

    for i in range(len(predict)):
        predict[i] = re.sub("[Label_]", "", predict[i])

    result = pd.DataFrame(predict, columns=["PREDICT"])
    result = pd.concat([sample_data["id"], result], axis=1)
    result.to_csv("./output/result_old.csv", index=False, encoding="utf-8", header=None)
    print("Done!\n")
