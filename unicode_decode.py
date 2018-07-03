import codecs

str1 = r'"\u6807\u7b7e": "\u673a\u5668\u4f5c\u8005"'
print(codecs.decode(str1, 'unicode_escape'))

str2 = r'"\u6807\u7b7e": "\u4eba\u7c7b\u4f5c\u8005"'
print(codecs.decode(str2, 'unicode_escape'))

str3 = r'"\u6807\u7b7e": "\u81ea\u52a8\u6458\u8981"'
print(codecs.decode(str3, 'unicode_escape'))

str4 = r'"\u6807\u7b7e": "\u673a\u5668\u7ffb\u8bd1"'
print(codecs.decode(str4, 'unicode_escape'))

with open(r"C:\Users\Administrator\Desktop\SMP-EUPT\training_data\training.txt", 'r') as f1, \
        open('training_chinese.txt', 'w', encoding='utf-8', errors='ignore') as f2:
    for line in f1:
        s = codecs.decode(line, 'unicode_escape', errors='ignore')
        f2.writelines(s)
