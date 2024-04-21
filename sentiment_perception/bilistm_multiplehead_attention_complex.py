import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import csv
import pymysql

def remove_links_mentions(tweet):
    link_re_pattern = "https?:\/\/t.co/[\w]+"
    mention_re_pattern = "@\w+"
    tweet = re.sub(link_re_pattern, "", tweet)
    tweet = re.sub(mention_re_pattern, "", tweet)
    return tweet.lower()


batch_size = 128


class BiLSTM_MHSattention_SentimentAnalysis(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout):
        super().__init__()

        # The embedding layer takes the vocab size and the embeddings size as input
        # The embeddings size is up to you to decide, but common sizes are between 50 and 100.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.nhidden=hidden_dim
        # The LSTM layer takes in the the embedding size and the hidden vector size.
        # The hidden dimension is up to you to decide, but common values are 32, 64, 128
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(2*hidden_dim, num_heads = 8,batch_first=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 8192, 1024)  # As bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(256, 32)
        self.fc4 = nn.Linear(32, 3)



    def forward(self, x):
        """
        The forward method takes in the input and the previous hidden state
        """

        # The input is transformed to embeddings by passing it to the embedding layer
        embs = self.embedding(x)
        embs = embs.permute(1, 0, 2)  # input : [len_seq, batch_size, embedding_dim]
        # The embedded inputs are fed to the LSTM alongside the previous hidden state
        hidden_state = torch.zeros(1 * 2, len(x), self.nhidden).to(device)
        cell_state = torch.zeros(1 * 2, len(x), self.nhidden).to(device)
        out, (final_hidden_state, final_cell_state) = self.lstm(embs, (hidden_state,cell_state))

        #out = self.dropout(out)
        out = out.permute(1, 0, 2)
        atten_out, attention = self.mha(out, out, out)
        atten_out = self.flatten(atten_out)
        out = self.fc1(atten_out)
        out = F.relu(out)
        # Dropout
        out = self.dropout(out)
        # Layer 2
        out = self.fc2(out)
        out = F.relu(out)
        # Layer 3
        out = self.fc3(out)
        out = F.relu(out)
        # Output layer
        out = self.fc4(out)
        # Dropout is applied to the output and fed to the FC layer
        #out = self.dropout(out)
        #out = out.permute(1, 0, 2)
        #attn_output, attn_output_weights = self.mha(out, out, out)
        # Flatten:
        #out = self.flatten(attn_output)
        #out = self.fc1(out)

        return out

train_path = "twitter_training.csv"
train_df = pd.read_csv(train_path)
train_df = train_df.drop(columns=["Id"])
train_df = train_df.drop(columns=["Pc"])
train_df = train_df.dropna()
train_df = train_df[train_df['Se'] != "Irrelevant"]
train_df = train_df[train_df['Co'] != "Not Available"]
train_df['Se'].value_counts()
train_clean_df, test_clean_df = train_test_split(train_df, test_size=0.15)

train_set = list(train_clean_df.to_records(index=False))
test_set = list(test_clean_df.to_records(index=False))
train_set = [(label, word_tokenize(remove_links_mentions(tweet))) for label, tweet in train_set]
test_set = [(label, word_tokenize(remove_links_mentions(tweet))) for label, tweet in test_set]

index2word = ["<PAD>", "<SOS>", "<EOS>"]

for ds in [train_set, test_set]:
    for label, tweet in ds:
        for token in tweet:
            if token not in index2word:
                index2word.append(token)

word2index = {token: idx for idx, token in enumerate(index2word)}


def label_map(label):
    if label == "Negative":
        return 0
    elif label == "Neutral":
        return 1
    else:  # Positive
        return 2


seq_length = 128


def encode_and_pad(tweet, length):
    sos = [word2index["<SOS>"]]
    eos = [word2index["<EOS>"]]
    pad = [word2index["<PAD>"]]

    if len(tweet) < length - 2:  # -2 for SOS and EOS
        n_pads = length - 2 - len(tweet)
        encoded = [word2index[w] for w in tweet]
        return sos + encoded + eos + pad * n_pads
    else:  # tweet is longer than possible; truncating
        encoded = [word2index[w] for w in tweet]
        truncated = encoded[:length - 2]
        return sos + truncated + eos


train_encoded = [(encode_and_pad(tweet, seq_length), label_map(label)) for label, tweet in train_set]
test_encoded = [(encode_and_pad(tweet, seq_length), label_map(label)) for label, tweet in test_set]

train_x = np.array([tweet for tweet, label in train_encoded])
train_y = np.array([label for tweet, label in train_encoded])
test_x = np.array([tweet for tweet, label in test_encoded])
test_y = np.array([label for tweet, label in test_encoded])

train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BiLSTM_MHSattention_SentimentAnalysis(len(word2index), 200, 64, 0.2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

epochs = 300
losses_train = []
batch_acc_train = []
epoch_loss_train = 0
model.train()

for e in range(epochs):
    for batch_idx, batch in enumerate(train_dl):
        input = batch[0].to(device)
        target = batch[1].to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            out = model(input)
            target = target.type_as(out)
            #out = out.reshape(out.shape[0], 1)
            loss = criterion(out, target.long())

            _, preds = torch.max(out,1)
            preds = preds.to("cpu").tolist()
            acc = accuracy_score(preds, target.tolist())

            loss.backward()
            optimizer.step()
            epoch_loss_train = loss.item()
            # epoch_acc += acc
            batch_acc_train.append(acc)
    losses_train.append(loss.item())
    print(
            f'Epoch: {e + 1:02}, Train Loss: {epoch_loss_train:.3f}, Train Acc: {sum(batch_acc_train) / len(batch_acc_train) * 100:.2f}%')
    # print(sum(batch_acc_train) / len(batch_acc_train))
plt.plot(losses_train)

data = pd.read_csv(r'datacsvdatong.csv', header=None,
                   names=['Id', 'title', 'tag1', 'tag2', 'tag3', 'tag4', 'tag5'])
data.to_csv('datacsvdatongnew.csv', index=False)
real_path = "datacsvdatongnew.csv"

# test_path = "twitter_validation_new.csv"

real_df = pd.read_csv(real_path)
# test_df = pd.read_csv(test_path)

real_df = real_df.drop(columns=["Id"])
# real_df = real_df.dropna()
real_df = real_df[real_df['title'] != "Irrelevant"]
# real_df = real_df[real_df['tags'] != "Not Available"]
# print(real_df)
real_set = list(real_df.to_records(index=False))

train_set2 = []
for title, tag1, tag2, tag3, tag4, tag5 in real_set:
    s = str(title) + str(tag1) + str(tag2) + str(tag3) + str(tag4) + str(tag5)
    train_set2.append(word_tokenize(remove_links_mentions(s)))

index2word2 = ["<PAD>", "<SOS>", "<EOS>"]

for ds in [train_set2]:
    for tweet in ds:
        for token in tweet:
            if token not in index2word2:
                index2word2.append(token)

word2index2 = {token: idx for idx, token in enumerate(index2word2)}


def encode_and_pad2(tweet, length):
    sos = [word2index2["<SOS>"]]
    eos = [word2index2["<EOS>"]]
    pad = [word2index2["<PAD>"]]

    if len(tweet) < length - 2:  # -2 for SOS and EOS
        n_pads = length - 2 - len(tweet)
        encoded = [word2index2[w] for w in tweet]
        return sos + encoded + eos + pad * n_pads
    else:  # tweet is longer than possible; truncating
        encoded = [word2index2[w] for w in tweet]
        truncated = encoded[:length - 2]
        return sos + truncated + eos


train_encoded2 = [(encode_and_pad2(tweet, seq_length)) for tweet in train_set2]

real_x = np.array([tweet for tweet in train_encoded2])
real_ds = TensorDataset(torch.from_numpy(real_x))
real_dl = DataLoader(real_ds, shuffle=True, batch_size=batch_size, drop_last=True)



idlist = []
with open('datacsvdatong.csv', 'r', encoding="utf-8") as f:
    reader = csv.reader(f)
    for row in reader:
        id = row[0]
        idlist.append(id)
    f.close()

model.eval()

conn = pymysql.connect(host='127.0.0.1',
                       port=3306,
                       user='root',
                       password='241033',
                       db='datongresult', charset='utf8')
cursor = conn.cursor()

idnum=0
emotion=""
for batch_idx, batch in enumerate(real_dl):
    input = batch[0].to(device)

    optimizer.zero_grad()
    with torch.set_grad_enabled(False):
        out = model(input, )
        _, preds = torch.max(out, 1)
        preds = preds.to("cpu").tolist()
        for i in preds:
            # print('预测类别为 => ', i)
            if i == 2:

                emotion="1"
            elif i == 1:

                emotion="2"
            else:

                emotion = "3"
            sql = "update datong_ugc set emotion = %s where id = %s"
            cursor.execute(sql, (emotion, idlist[idnum]))
            conn.commit()
            idnum = idnum + 1

conn.close()








