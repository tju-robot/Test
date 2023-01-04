import re
import random
import tarfile
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from model import RNN, LSTM, GRU


def load_imdb(is_training):
    data_set = []
    for label in ["pos", "neg"]:
        with tarfile.open("./aclImdb_v1.tar.gz") as tarf:
            path_pattern = "aclImdb/train/" + label + "/.*\.txt$" if is_training \
                else "aclImdb/test/" + label + "/.*\.txt$"
            path_pattern = re.compile(path_pattern)  # 生成正则对象
            tf = tarf.next()  # 显示下一个文件信息
            while tf != None:
                if bool(path_pattern.match(tf.name)):
                    sentence = tarf.extractfile(tf).read().decode()  # 从tf文件中提取文件，读取内容并解码
                    sentence_label = 0 if label == 'neg' else 1
                    data_set.append((sentence, sentence_label))
                tf = tarf.next()
    return data_set


def data_preprocess(corpus):
    data_set = []
    for sentence, sentence_label in corpus:
        # 这里是把所有的句子转换为小写，从而减小词表的大小
        sentence = sentence.strip().lower()
        sentence = sentence.split(" ")  # 使用空格分词

        data_set.append((sentence, sentence_label))

    return data_set


# 构造词典
def build_dict(corpus):
    word_freq_dict = dict()
    for sentence, _ in corpus:
        for word in sentence:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)  # 排序

    word2id_dict = dict()  # word 2 id
    word2id_freq = dict()  # id 2 freq

    # 一般来说，我们把oov和pad放在词典前面，给他们一个比较小的id，这样比较方便记忆，并且易于后续扩展词表
    word2id_dict['[oov]'] = 0
    word2id_freq[0] = 1e10

    word2id_dict['[pad]'] = 1
    word2id_freq[1] = 1e10

    for word, freq in word_freq_dict:
        word2id_dict[word] = len(word2id_dict)  # 频率最大的单词，id越小
        word2id_freq[word2id_dict[word]] = freq

    return word2id_freq, word2id_dict


# 把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    data_set = []
    for sentence, sentence_label in corpus:
        # 将句子中的词逐个替换成id，如果句子中的词不在词表内，则替换成oov
        sentence = [word2id_dict[word] if word in word2id_dict else word2id_dict['[oov]'] for word in sentence]
        data_set.append((sentence, sentence_label))
    return data_set


# 编写一个迭代器，每次调用这个迭代器都会返回一个新的batch，用于训练或者预测
def build_batch(word2id_dict, corpus, batch_size, epoch_num,
                max_seq_len, shuffle=True, drop_last=True):
    # 模型将会接受的两个输入：
    # 1. 一个形状为[batch_size, max_seq_len]的张量，sentence_batch，代表了一个mini-batch的句子。
    # 2. 一个形状为[batch_size, 1]的张量，sentence_label_batch，每个元素都是非0即1，代表了每个句子的情感类别（正向或者负向）
    sentence_batch = []
    sentence_label_batch = []

    for _ in range(epoch_num):
        # 每个epoch前都shuffle一下数据，有助于提高模型训练的效果
        if shuffle:
            random.shuffle(corpus)

        for sentence, sentence_label in corpus:
            sentence_sample = sentence[:min(max_seq_len, len(sentence))]
            if len(sentence_sample) < max_seq_len:
                for _ in range(max_seq_len - len(sentence_sample)):
                    sentence_sample.append(word2id_dict['[pad]'])

            sentence_sample = [[word_id] for word_id in sentence_sample]

            sentence_batch.append(sentence_sample)
            sentence_label_batch.append([sentence_label])

            if len(sentence_batch) == batch_size:
                yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")
                sentence_batch = []
                sentence_label_batch = []
        if not drop_last and len(sentence_batch) > 0:
            yield np.array(sentence_batch).astype("int64"), np.array(sentence_label_batch).astype("int64")


def SoftCrossEntropy(inputs, target, reduction='sum'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


# 定义一个用于情感分类的网络实例
class SentimentClassifier(torch.nn.Module):

    def __init__(self, hidden_size, vocab_size, embedding_size, class_num=2, num_steps=128, num_layers=1,
                 init_scale=0.1, dropout_rate=None):
        # hidden_size，表示embedding-size，hidden和cell向量的维度 vocab_size，模型可以考虑的词表大小
        # embedding_size，表示词向量的维度 class_num，情感类型个数，可以是2分类，也可以是多分类
        # num_steps，表示这个情感分析模型最大可以考虑的句子长度  num_layers，表示网络的层数
        # dropout_rate，表示使用dropout过程中失活的神经元比例
        # init_scale，表示网络内部的参数的初始化范围,长短时记忆网络内部用了很多Tanh，Sigmoid等激活函数
        super(SentimentClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.class_num = class_num
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale

        # 声明一个循环神经网络模型，用来把每个句子抽象成向量
        self.simple_lstm_rnn = RNN(input_size=hidden_size, hidden_size=hidden_size, output_size=num_layers)
        # 声明一个embedding层，用来把句子中的每个词转换为向量
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size, sparse=False)
        # 声明使用上述语义向量映射到具体情感类别时所需要使用的线性层
        self.cls_fc = torch.nn.Linear(in_features=self.hidden_size, out_features=self.num_steps)
        # 添加的全连接层
        self.cls_fc1 = torch.nn.Linear(in_features=self.num_steps, out_features=self.class_num)
        # 一般在获取单词的embedding后，会使用dropout层，防止过拟合，提升模型泛化能力
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate)

    def forward(self, inputs):
        # input为输入的训练文本，其shape为[batch_size, max_seq_len]
        batch_size = inputs.shape[0]

        # 首先我们需要定义LSTM的初始hidden和cell，这里我们使用0来初始化这个序列的记忆
        init_hidden_data = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype='float32')
        # 设置stop_gradient=True，避免这些向量被更新，从而影响训练效果
        init_hidden = torch.tensor(init_hidden_data).to("cuda")
        init_hidden.stop_gradient = True
        x_emb = self.embedding(inputs)
        x_emb = torch.reshape(x_emb, shape=[-1, self.num_steps, self.embedding_size])
        # 在获取的词向量后添加dropout层
        x_emb = self.dropout_layer(x_emb)
        # 使用循环神经网络网络，把每个句子转换为语义向量
        rnn_out, last_hidden = self.simple_lstm_rnn(x_emb, init_hidden)
        # 提取最后一层隐状态作为文本的语义向量
        last_hidden = torch.reshape(last_hidden[-1], shape=[-1, self.hidden_size])
        preds_1 = self.cls_fc(last_hidden)
        preds = self.cls_fc1(preds_1)

        return preds


def main():
    train_corpus = load_imdb(True)
    test_corpus = load_imdb(False)
    train_corpus = data_preprocess(train_corpus)
    test_corpus = data_preprocess(test_corpus)
    word2id_freq, word2id_dict = build_dict(train_corpus)
    vocab_size = len(word2id_freq)
    train_corpus = convert_corpus_to_id(train_corpus, word2id_dict)
    test_corpus = convert_corpus_to_id(test_corpus, word2id_dict)
    # 定义训练参数
    epoch_num = 10
    batch_size = 128

    learning_rate = 0.01
    dropout_rate = 0.1
    num_layers = 1
    hidden_size = 256
    embedding_size = 256
    max_seq_len = 128

    # 实例化模型
    sentiment_classifier = SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                               num_layers=num_layers, dropout_rate=dropout_rate)

    # 指定优化策略，更新模型参数
    optimizer = torch.optim.Adam(params=sentiment_classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    # 定义训练函数
    # 记录训练过程中的损失变化情况，可用于后续画图查看训练情况
    losses = []
    steps = []

    def train(model):
        # 开启模型训练模式
        model.cuda()

        # 建立训练数据生成器，每次迭代生成一个batch，每个batch包含训练文本和文本对应的情感标签
        train_loader = build_batch(word2id_dict, train_corpus, batch_size, epoch_num, max_seq_len)

        for step, (sentences, labels) in enumerate(train_loader):
            # 获取数据，并将张量转换为Tensor类型
            sentences = torch.tensor(sentences)

            labels = torch.tensor(labels)
            # 前向计算，将数据feed进模型，并得到预测的情感标签和损失
            sentences, labels = sentences.to("cuda"), labels.to("cuda")
            output = model(sentences)
            output = output.to(torch.float32)
            labels = labels.to(torch.float32)
            # 计算损失
            loss = SoftCrossEntropy(inputs=output, target=labels)

            loss = torch.mean(loss)
            loss = loss.to(torch.float32)

            # 后向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 清除梯度
            optimizer.zero_grad()
            if step % 100 == 0:
                # 记录当前步骤的loss变化情况
                losses.append(loss.data)
                steps.append(step)
                # 打印当前loss数值
                print("step %d, loss %.3f" % (step, loss.data))

    # 训练模型
    train(sentiment_classifier)

    # 保存模型，包含两部分：模型参数和优化器参数
    model_name = "sentiment_classifier_RNN"
    # 保存训练好的模型参数
    torch.save(sentiment_classifier.state_dict(), "{}.modelpkl".format(model_name))
    # 保存优化器参数，方便后续模型继续训练
    torch.save(optimizer.state_dict(), "{}.optimizerpkl".format(model_name))

    def evaluate(model):
        # 开启模型测试模式，在该模式下，网络不会进行梯度更新
        model.eval()

        # 定义以上几个统计指标
        tp, tn, fp, fn = 0, 0, 0, 0

        # 构造测试数据生成器
        test_loader = build_batch(word2id_dict, test_corpus, batch_size, 1, max_seq_len)
        for sentences, labels in test_loader:
            sentences = torch.tensor(sentences)
            labels = torch.tensor(labels)
            sentences, labels = sentences.to("cuda"), labels.to("cuda")
            # 获取模型对当前batch的输出结果
            output = model(sentences)

            # 使用softmax进行归一化
            probs = F.softmax(output)

            # 把输出结果转换为numpy array数组，比较预测结果和对应label之间的关系，并更新tp，tn，fp和fn
            probs = probs.cpu().detach().numpy()
            for i in range(len(probs)):
                # 当样本是的真实标签是正例
                if labels[i][0] == 1:
                    # 模型预测是正例
                    if probs[i][1] > probs[i][0]:
                        tp += 1
                    # 模型预测是负例
                    else:
                        fn += 1
                # 当样本的真实标签是负例
                else:
                    # 模型预测是正例
                    if probs[i][1] > probs[i][0]:
                        fp += 1
                    # 模型预测是负例
                    else:
                        tn += 1

        # 整体准确率
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        a = tp + tn + fp + fn
        true = []
        pre = []
        for i in range(a):
            if i < tp:
                true.append(1)
                pre.append(1)
            elif i < tp + fp:
                true.append(0)
                pre.append(1)
            elif i < tp + fp + fn:
                true.append(1)
                pre.append(0)
            else:
                true.append(0)
                pre.append(0)

        # 绘制混淆矩阵
        cm = confusion_matrix(true, pre)
        cm_display = ConfusionMatrixDisplay(cm)
        cm_display.plot()
        plt.show()
        # 输出最终评估的模型效果
        print("TP: {}\nFP: {}\nTN: {}\nFN: {}\n".format(tp, fp, tn, fn))
        print("Accuracy: %.4f" % accuracy)

    # 加载训练好的模型进行预测，重新实例化一个模型，然后将训练好的模型参数加载到新模型里面
    saved_state = torch.load("./sentiment_classifier_RNN.modelpkl")
    sentiment_classifier = SentimentClassifier(hidden_size, vocab_size, embedding_size, num_steps=max_seq_len,
                                               num_layers=num_layers, dropout_rate=dropout_rate)
    sentiment_classifier.load_state_dict(saved_state)
    sentiment_classifier.to("cuda")
    # 评估模型
    evaluate(sentiment_classifier)


if __name__ == "__main__":
        main()