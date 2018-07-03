import sys
import json
import jieba
import codecs
import importlib

importlib.reload(sys)

with open(r"C:\Users\Administrator\Desktop\SMP-EUPT\validation_data\validation.txt", 'r') as f1, \
        open('validation_split.txt', 'w', encoding='utf-8') as f2:
    for line in f1.readlines():
        dict = json.loads(line)

        number = dict["id"]
        content = dict["内容"]

        seg_list = jieba.cut(content)

        outline = "\t" + "内容:" + " " + " ".join(seg_list)
        outline = "id:" + str(number) + outline + " "
        f2.write(outline)

print("\nWord segmentation complete.")
