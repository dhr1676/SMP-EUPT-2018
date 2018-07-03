import fasttext
import csv
import json
import jieba

classifier = fasttext.load_model('model.bin', label_prefix='__label__')

with open(r"C:\Users\Administrator\Desktop\SMP-EUPT\validation_data\validation.txt", 'r', encoding='utf-8',
          errors='ignore') as f1, open('result.csv', 'w', encoding='utf-8', errors='ignore', newline='') as f2:
    writer = csv.writer(f2)
    for line in f1.readlines():
        dic = json.loads(line)

        id = dic['id']
        content = dic["内容"].strip()

        seg_content = ' '.join(jieba.cut(content.replace("\t", " ").replace("\n", " ")))

        li = content.split("$$$$$$$$")
        lbl = classifier.predict(li)

        writer.writerow([id, str(lbl)[3:-3]])
