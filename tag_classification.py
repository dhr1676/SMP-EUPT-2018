import chardet
import codecs


def articalClassify(line):
    str1 = r'"\u6807\u7b7e": "\u81ea\u52a8\u6458\u8981"'  # 自动摘要
    str2 = r'"\u6807\u7b7e": "\u673a\u5668\u7ffb\u8bd1"'  # 机器翻译
    str3 = r'"\u6807\u7b7e": "\u673a\u5668\u4f5c\u8005"'  # 机器作者
    str4 = r'"\u6807\u7b7e": "\u4eba\u7c7b\u4f5c\u8005"'  # 人类作者
    if str1 in line:
        return 1
    elif str2 in line:
        return 2
    elif str3 in line:
        return 3
    elif str4 in line:
        return 4


with open(r"C:\Users\Administrator\Desktop\SMP-EUPT\training_data\training.txt", 'r') as f1, \
        open('auto_abstract.txt', 'w', encoding='utf-8', errors='ignore') as f2, \
        open('machine_translate.txt', 'w', encoding='utf-8', errors='ignore') as f3, \
        open('machine_author.txt', 'w', encoding='utf-8', errors='ignore') as f4, \
        open('human_author.txt', 'w', encoding='utf-8', errors='ignore') as f5:
    for line in f1:
        s = codecs.decode(line, 'unicode_escape', errors='ignore')
        print(articalClassify(line))
        if articalClassify(line) == 1:
            f2.writelines(s)
        elif articalClassify(line) == 2:
            f3.writelines(s)
        elif articalClassify(line) == 3:
            f4.writelines(s)
        elif articalClassify(line) == 4:
            f5.writelines(s)
