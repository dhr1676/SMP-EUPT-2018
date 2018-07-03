str1 = r'"\u81ea\u52a8\u6458\u8981"'
str2 = r'"\u673a\u5668\u7ffb\u8bd1"'
str3 = r'"\u673a\u5668\u4f5c\u8005"'
str4 = r'"\u4eba\u7c7b\u4f5c\u8005"'

# The first r 防止转义
# The second 'r' means open file mode,
# 'r'：只读（缺省。如果文件不存在，则抛出错误）
# 'w'：只写（如果文件不存在，则自动创建文件）
# 'a'：附加到文件末尾
# 'r+'：读写
with open(r"./input/training_new.txt", 'r') as f:
    s = f.read()

    print("\u81ea\u52a8\u6458\u8981")
    print(s.count(str1))

    print("\u673a\u5668\u7ffb\u8bd1")
    print(s.count(str2))

    print("\u673a\u5668\u4f5c\u8005")
    print(s.count(str3))

    print("\u4eba\u7c7b\u4f5c\u8005")
    print(s.count(str4))

"""
自动摘要: 31034
机器翻译: 36206
机器作者: 31163
人类作者: 48018
"""