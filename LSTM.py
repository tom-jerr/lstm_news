## 导入本章所需要的模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
## 输出图显示中文
from matplotlib.font_manager import FontProperties

fonts = FontProperties(fname="/Library/Fonts/华文细黑.ttf")
import re
import string
import copy
import time
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import jieba
from torchtext import data
from torchtext.vocab import Vectors


## 读取训练、验证和测试数据集
train_df = pd.read_csv("cnews/cnews.train.txt", sep="\t",
                       header=None, names=["label", "text"])
val_df = pd.read_csv("cnews/cnews.val.txt", sep="\t",
                     header=None, names=["label", "text"])
test_df = pd.read_csv("cnews/cnews.test.txt", sep="\t",
                      header=None, names=["label", "text"])
stop_words = pd.read_csv("cnews/stopwords.txt",
                         header=None, names=["text"])


def chinese_pre(text_data):
    ## 字母转化为小写,去除数字,
    text_data = text_data.lower()
    text_data = re.sub("\d+", "", text_data)
    ## 分词,使用精确模式
    text_data = list(jieba.cut(text_data, cut_all=False))
    ## 去停用词和多余空格
    text_data = [word.strip() for word in text_data if word not in stop_words.text.values]
    ## 处理后的词语使用空格连接为字符串
    text_data = " ".join(text_data)
    return text_data


## 对数据进行分词
train_df["cutword"] = train_df.text.apply(chinese_pre)
val_df["cutword"] = val_df.text.apply(chinese_pre)
test_df["cutword"] = test_df.text.apply(chinese_pre)

## 预处理后的结果保存为新的文件
train_df[["label", "cutword"]].to_csv("data/chap7/cnews_train.csv", index=False)
val_df[["label", "cutword"]].to_csv("data/chap7/cnews_val.csv", index=False)
test_df[["label", "cutword"]].to_csv("data/chap7/cnews_test.csv", index=False)
train_df = pd.read_csv("data/chap7/cnews_train.csv")
val_df = pd.read_csv("data/chap7/cnews_val.csv")
test_df = pd.read_csv("data/chap7/cnews_test.csv")
labelMap = {"体育": 0, "娱乐": 1, "家居": 2, "房产": 3, "教育": 4,
            "时尚": 5, "时政": 6, "游戏": 7, "科技": 8, "财经": 9}
train_df["labelcode"] = train_df["label"].map(labelMap)
val_df["labelcode"] = val_df["label"].map(labelMap)
test_df["labelcode"] = test_df["label"].map(labelMap)

train_df[["labelcode", "cutword"]].to_csv("data/chap7/cnews_train2.csv", index=False)
val_df[["labelcode", "cutword"]].to_csv("data/chap7/cnews_val2.csv", index=False)
test_df[["labelcode", "cutword"]].to_csv("data/chap7/cnews_test2.csv", index=False)
# # 使用torchtext库进行数据准备
# 定义文件中对文本和标签所要做的操作
# """
# sequential=True:表明输入的文本时字符，而不是数值字
# tokenize="spacy":使用spacy切分词语
# use_vocab=True: 创建一个词汇表
# batch_first=True: batch优先的数据方式
# fix_length=400 :每个句子固定长度为400
# """
## 定义文本切分方法，因为前面已经做过处理，所以直接使用空格切分即可
mytokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=mytokenize,
                  include_lengths=True, use_vocab=True,
                  batch_first=True, fix_length=400)
LABEL = data.Field(sequential=False, use_vocab=False,
                   pad_token=None, unk_token=None)
## 对所要读取的数据集的列进行处理
text_data_fields = [
    ("labelcode", LABEL),  # 对标签的操作
    ("cutword", TEXT)  # 对文本的操作
]
## 读取数据
traindata, valdata, testdata = data.TabularDataset.splits(
    path="data/chap7", format="csv",
    train="cnews_train2.csv", fields=text_data_fields,
    validation="cnews_val2.csv",
    test="cnews_test2.csv", skip_header=True
)
## 使用训练集构建单词表,没有预训练好的词项量
TEXT.build_vocab(traindata, max_size=20000, vectors=None)
LABEL.build_vocab(traindata)
# # ## 可视化训练集中的前50个高频词
# # word_fre = TEXT.vocab.freqs.most_common(n=50)
# # word_fre = pd.DataFrame(data=word_fre, columns=["word", "fre"])
# # word_fre.plot(x="word", y="fre", kind="bar", legend=False, figsize=(12, 7))
# # plt.xticks(rotation=90, fontproperties=fonts, size=10)
# # plt.show()
# #
# # print("词典的词数:", len(TEXT.vocab.itos))
# # print("前10个单词:\n", TEXT.vocab.itos[0:10])
# # ## 类别标签的数量和类别
# # print("类别标签情况:", LABEL.vocab.freqs)
## 定义一个迭代器，将类似长度的示例一起批处理。
BATCH_SIZE = 64
train_iter = data.BucketIterator(traindata, batch_size=BATCH_SIZE)
val_iter = data.BucketIterator(valdata, batch_size=BATCH_SIZE)
test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE)
# ##  获得一个batch的数据，对数据进行内容进行介绍
# for step, batch in enumerate(train_iter):
#     if step > 0:
#         break
# ## 针对一个batch 的数据，可以使用batch.labelcode获得数据的类别标签
# print("数据的类别标签:\n", batch.labelcode)
# ## batch.cutword[0]是文本对应的标签向量
# print("数据的尺寸:", batch.cutword[0].shape)
# ## batch.cutword[1] 对应每个batch使用的原始数据中的索引
# print("数据样本数:", len(batch.cutword[1]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        """
        vocab_size:词典长度
        embedding_dim:词向量的维度
        hidden_dim: RNN神经元个数
        layer_dim: RNN的层数
        output_dim:隐藏层输出的维度(分类的数量)
        """
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim  ## RNN神经元个数
        self.layer_dim = layer_dim  ## RNN的层数
        ## 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM及全连接层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, layer_dim,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.lstm(embeds, None)  # None 表示 hidden state 会用全0的 state
        # 选取最后一个时间点的out输出
        out = self.fc1(r_out[:, -1, :])
        return out
# 输入数据格式： （三个输入）
# input(seq_len, batch, input_size)
# h_0(num_layers * num_directions, batch, hidden_size)
# c_0(num_layers * num_directions, batch, hidden_size)
# 输出数据格式：
# output(seq_len, batch, hidden_size * num_directions)
# h_n(num_layers * num_directions, batch, hidden_size)
# c_n(num_layers * num_directions, batch, hidden_size)
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 128
layer_dim = 1
output_dim = 10
lstmmodel = LSTMNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)
print(lstmmodel)


## 定义网络的训练过程函数
def train_model2(model, traindataloader, valdataloader, criterion,
                 optimizer, num_epochs=25, ):
    """
    model:网络模型；traindataloader:训练数据集;
    valdataloader:验证数据集，;criterion：损失函数；optimizer：优化方法；
    num_epochs:训练的轮数
    """
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    # since = time.time()
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0.0
        val_corrects = 0
        val_num = 0
        model.train()  ## 设置模型为训练模式
        for step, batch in enumerate(traindataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)  # 预测的标签
            loss = criterion(out, target)  # 计算损失函数值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num += len(target)
        ## 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        ## 计算一个epoch的训练后在验证集上的损失和精度
        model.eval()  ## 设置模型为训练模式评估模式
        for step, batch in enumerate(valdataloader):
            textdata, target = batch.cutword[0], batch.labelcode.view(-1)
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, target)
            val_loss += loss.item() * len(target)
            val_corrects += torch.sum(pre_lab == target.data)
            val_num += len(target)
        ## 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{} Val Loss: {:.4f}  Val Acc: {:.4f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return model, train_process


# 定义优化器
optimizer = torch.optim.Adam(lstmmodel.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss()  # 损失函数
## 对模型进行迭代训练,对所有的数据训练EPOCH轮
lstmmodel, train_process = train_model2(
    lstmmodel, train_iter, val_iter, loss_func, optimizer, num_epochs=20)
## 输出结果保存和数据保存
torch.save(lstmmodel, "data/chap7/lstmmodel.pkl")

# 导入保存的模型
# lstmmodel = torch.load("data/chap7/lstmmodel.pkl")
# lstmmodel
# ## 保存训练过程
# train_process.to_csv("data/chap7/lstmmodel_process.csv", index=False)
# train_process

## 可视化模型训练过程中
# plt.figure(figsize=(18, 6))
# plt.subplot(1, 2, 1)
# plt.plot(train_process.epoch, train_process.train_loss_all,
#          "r.-", label="Train loss")
# plt.plot(train_process.epoch, train_process.val_loss_all,
#          "bs-", label="Val loss")
# plt.legend()
# plt.xlabel("Epoch number", size=13)
# plt.ylabel("Loss value", size=13)
# plt.subplot(1, 2, 2)
# plt.plot(train_process.epoch, train_process.train_acc_all,
#          "r.-", label="Train acc")
# plt.plot(train_process.epoch, train_process.val_acc_all,
#          "bs-", label="Val acc")
# plt.xlabel("Epoch number", size=13)
# plt.ylabel("Acc", size=13)
# plt.legend()
# plt.show()
# 对测试集进行预测并计算精度
lstmmodel.eval()  ## 设置模型为训练模式评估模式
test_y_all = torch.LongTensor()
pre_lab_all = torch.LongTensor()
for step, batch in enumerate(test_iter):
    textdata, target = batch.cutword[0], batch.labelcode.view(-1)
    out = lstmmodel(textdata)
    pre_lab = torch.argmax(out, 1)
    test_y_all = torch.cat((test_y_all, target))  ##测试集的标签
    pre_lab_all = torch.cat((pre_lab_all, pre_lab))  ##测试集的预测标签

acc = accuracy_score(test_y_all, pre_lab_all)

## 计算混淆矩阵并可视化
class_label = ["体育", "娱乐", "家居", "房产", "教育",
               "时尚", "时政", "游戏", "科技", "财经"]
conf_mat = confusion_matrix(test_y_all, pre_lab_all)
df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                             ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                             ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
# plt.figure(figsize=(10, 7))
# heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
# heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
#                              ha='right')
# heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
#                              ha='right')
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()




















