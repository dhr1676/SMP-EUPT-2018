import sys
import json
import jieba
import codecs
import importlib

importlib.reload(sys)

with open(r"training.txt", 'r') as f1, \
        open('training_split.txt', 'w', encoding='utf-8') as f2:
    for line in f1.readlines():
        dict = json.loads(line)

        lable = dict["标签"]
        content = dict["内容"].strip()

        seg_list = jieba.cut(content.replace("\t", " ").replace("\n", " "))

        outline = "\t" + " ".join(seg_list)
        outline = "__label__" + lable + outline + "\n"
        f2.write(outline)

print("\nWord segmentation complete.")
