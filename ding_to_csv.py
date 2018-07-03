import pandas as pd
import json

with open(r"./input/validation.txt", 'r') as f1:
    # readlines will return a list, list length is line's number of file
    dct = f1.readlines()

    label_list = []
    content_list = []

    for i in range(len(dct)):
        line = json.loads(dct[i])
        # label_list.append(line['标签'])
        label_list.append(line['id'])
        content_list.append(line['内容'])

    # df = pd.DataFrame(label_list, columns=["Label"])
    df = pd.DataFrame(label_list, columns=["id"])
    df = pd.concat([df, pd.DataFrame(content_list, columns=["Content"])], axis=1)

    print(df.shape)

    df.to_csv("./output/validation.csv", index=False, encoding="utf-8")
    print("Writing to validation.csv\n")
