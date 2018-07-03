import fasttext

# classifier = fasttext.supervised("training_split.txt","model",label_prefix="__label__")
classifier = fasttext.supervised("trainning_split.txt", "model",
                                 label_prefix="__label__",
                                 lr=0.3,
                                 epoch=100,
                                 dim=200,
                                 bucket=5000000)
