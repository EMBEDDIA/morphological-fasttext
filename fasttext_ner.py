import configparser
import math
import os

import fasttext
import pandas as pd

import torch
import torch.nn as nn
from seqeval.metrics import classification_report
from torch.autograd import Variable

import numpy as np
from torch.nn import functional as F

from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, Dataset

# print(torch.cuda.is_available())
# print(next(model.parameters()).is_cuda)
# print(var_name.is_cuda)




class NERLSTM(nn.Module):
    # def __init__(self, nb_lstm_layers, fasttext_encoding, nb_lstm_units=100, embedding_dim=3, batch_size=3):
    def __init__(self, nb_lstm_layers, tags, device, sentence_dim, batch_size, nb_tags=7, nb_lstm_units=256, embedding_dim=300):
        super(NERLSTM, self).__init__()
        # self.fasttext_encoding = fasttext_encoding
        # self.vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8,
        #               'yeah': 9}
        # # self.tags = {'<PAD>': 0, 'LOC': 1, 'PER': 2, 'ORG': 3}
        # self.tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}

        # self.vocab = vocab
        self.tags = tags

        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        # don't count the padding tag for the classifier output
        self.nb_tags = nb_tags
        self.sentence_dim = sentence_dim

        self.on_gpu = False

        # when the model is bidirectional we double the output dimension
        # self.lstm
        self.device = device
        self.on_gpu = True
        self.dropout = nn.Dropout(0.1)

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first
        # nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        # padding_idx = self.vocab['<PAD>']
        # self.word_embedding = nn.Embedding(
        #     num_embeddings=nb_vocab_words,
        #     embedding_dim=self.embedding_dim,
        #     padding_idx=padding_idx
        # )

        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

        # create initial values of LSTM
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the weights are of the form (nb_lstm_layers, batch_size, nb_lstm_units)
        # hidden_a = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        # hidden_b = torch.randn(self.hparams.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if self.on_gpu:
            # hidden_a = hidden_a.cuda()
            # hidden_b = hidden_b.cuda()
            hidden_a = hidden_a.to(self.device)
            hidden_b = hidden_b.to(self.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths, upos=None, feats=None, fixes=None):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        # self.hidden = self.init_hidden()

        # a = X.size()
        # batch_size, seq_len = X.shape
        # batch_size, seq_len, _ = X.size()

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        # X = self.word_embedding(X)

        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM TODO x or X
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)

        # now run through LSTM
        X, _ = self.lstm(X, self.hidden)

        # undo the packing operation
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=self.sentence_dim)

        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        X = X.contiguous()
        X = X.view(-1, X.shape[2])

        X = self.dropout(X)
        # run through actual linear layer
        X = self.hidden_to_tag(X)

        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        # TODO COMMENT!!!
        # X = F.log_softmax(X, dim=1)

        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        # X = X.view(batch_size, seq_len, self.nb_tags)

        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y, X_lengths):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        Y = Y.view(-1)

        # a = np.zeros(([3, 7, 5]))
        # Y_hat = torch.zeros(3, 7, 5)

        # flatten all predictions
        Y_hat = Y_hat.view(-1, self.nb_tags)



        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()

        # count how many tokens we have
        nb_tokens = int(torch.sum(mask))
        # nb_tokens = int(torch.sum(mask).data[0])

        # a = range(Y_hat.shape[0])
        # b = Y - mask.long()
        # c = Y_hat[range(Y_hat.shape[0]), Y]

        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y - mask.long()] * mask
        # Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask

        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens

        return ce_loss




def readfile_ner(filename, cv_part):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    filename = filename % cv_part
    df = pd.read_csv(filename, sep='\t', keep_default_na=False)
    df = df.fillna('')
    # first_sentence_i = df['sentence_id']
    first_sentence_i = df['sentence_id'][0]
    # last_sentence_i = df['sentence_id'].tail(1)
    last_sentence_i = df['sentence_id'].tail(1).iloc[0]

    output = []

    for i in range(first_sentence_i, last_sentence_i):
        df_sentence = df.loc[df['sentence_id'] == i]
        sentence = []
        labels = []
        others = []
        for _, data in df_sentence.iterrows():
            if isinstance(data['word'], float):
                data['word'] = ''
            sentence.append(data['word'])
            if data['word'] == '"':
                continue
            other = {}
            if 'msd' in data:
                other['msd'] = data['msd']
            if 'upos' in data:
                other['upos'] = data['upos']
            if 'feats' in data:
                other['feats'] = data['feats']
            if 'xpos' in data:
                other['xpos'] = data['xpos']
            if 'lemma' in data:
                other['lemma'] = data['lemma']
            if 'dependency_relation' in data:
                other['dependency_relation'] = data['dependency_relation']
            if 'prefixes' in data:
                other['prefixes'] = data['prefixes']
            if 'suffixes' in data:
                other['suffixes'] = data['suffixes']

            others.append(other)
            # if data['label'] == float('nan'):
            if not isinstance(data['label'], str) and math.isnan(data['label']):
                labels.append('O')
            else:
                labels.append(data['label'])
        output.append((sentence, labels, others))

    return output







def train(padded_X, X_lengths, padded_Y, test_X, test_X_lengths, test_Y, label_map, model, batch_size, longest_sent, optimizer, criterion, device, nb_epoch=100, upos=None, feats=None, fixes=None, inside_eval=False):
    for epoch in range(nb_epoch):
        # train
        for example_i in range(0, len(padded_X), batch_size):
            # TODO Erase this
            # If last batch size != 16 break
            if example_i + batch_size > len(padded_X):
                break
            X_ids = padded_X[example_i:min(example_i + batch_size, len(padded_X))]
            if upos:
                upos_ids = upos[example_i:min(example_i + batch_size, len(upos))]
            if feats:
                feats_ids = feats[example_i:min(example_i + batch_size, len(feats))]
            if fixes:
                fixes_ids = fixes[example_i:min(example_i + batch_size, len(fixes))]
            X_leng = X_lengths[example_i:min(example_i + batch_size, len(X_lengths))]

            Y_ids = padded_Y[example_i:min(example_i + batch_size, len(padded_Y))]

            if upos:
                if feats:
                    if fixes:
                        sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids), key=lambda pair: pair[0],
                                             reverse=True)

                        X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = zip(*sorted_data)
                        X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(
                            upos_ids), list(feats_ids), list(fixes_ids)
                    else:
                        sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, feats_ids), key=lambda pair: pair[0], reverse=True)

                        X_leng, X_ids, Y_ids, upos_ids, feats_ids = zip(*sorted_data)
                        X_leng, X_ids, Y_ids, upos_ids, feats_ids = list(X_leng), list(X_ids), list(Y_ids), list(upos_ids), list(feats_ids)
                else:
                    sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids), key=lambda pair: pair[0], reverse=True)

                    X_leng, X_ids, Y_ids, upos_ids = zip(*sorted_data)
                    X_leng, X_ids, Y_ids, upos_ids = list(X_leng), list(X_ids), list(Y_ids), list(upos_ids)
            else:
                sorted_data = sorted(zip(X_leng, X_ids, Y_ids), key=lambda pair: pair[0], reverse=True)

                X_leng, X_ids, Y_ids = zip(*sorted_data)
                X_leng, X_ids, Y_ids = list(X_leng), list(X_ids), list(Y_ids)


            # X_leng, X_ids, Y_ids = sorted(zip(X_leng, X_ids, Y_ids))

            Y_ids = torch.tensor([index for exam in Y_ids for index in exam], dtype=torch.long) - 1
            Y_ids = Y_ids.to(device)

            X_ids = torch.tensor(X_ids, dtype=torch.float32)
            X_ids = X_ids.to(device)
            if upos:
                upos_ids = torch.tensor(upos_ids, dtype=torch.float32)
                upos_ids = upos_ids.to(device)
            if feats:
                feats_ids = torch.tensor(feats_ids, dtype=torch.float32)
                feats_ids = feats_ids.to(device)
            if fixes:
                fixes_ids = torch.tensor(fixes_ids, dtype=torch.float32)
                fixes_ids = fixes_ids.to(device)


            # Y_ids = torch.tensor(Y_ids, dtype=torch.long) - 1
    # padded_X_ids = torch.tensor(padded_X, dtype=torch.long)
    # padded_Y_ids = torch.tensor([index for exam in padded_Y for index in exam], dtype=torch.long) - 1


            # Forward pass: Compute predicted y by passing x to the model
            # y_pred = model(padded_X, X_lengths)
            # if not all(X_leng):
            #     print('here')
            if upos:
                y_pred = model(X_ids, X_leng, upos_ids)
            elif feats:
                y_pred = model(X_ids, X_leng, upos_ids, feats_ids)
            elif fixes:
                y_pred = model(X_ids, X_leng, upos_ids, feats_ids, fixes_ids)
            else:
                y_pred = model(X_ids, X_leng)
            # y_pred = model(new_padded_X_ids, X_lengths)

            # Compute and print loss
            # loss = model.loss(y_pred, padded_Y_ids, X_lengths)
            # loss = model.loss(y_pred, new_padded_Y_ids, X_lengths)
            # TODO CREATE THIS!!! down
            # criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
            # a = [index for exam in padded_Y_ids for index in exam]
            # loss = criterion(y_pred, [index for exam in padded_Y_ids for index in exam])
            # padded_Y_ids = torch.tensor([index for exam in padded_Y for index in exam], dtype=torch.long) - 1

            loss = criterion(y_pred, Y_ids)


            # classification_report(Y_ids, test)

            # loss = loss_fn(y_pred, padded_Y_ids)
            if example_i == 0:
                print(epoch, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # TODO ERASE
            # if example_i > 800:
            #     break


        if inside_eval:
            test(test_X, test_X_lengths, test_Y, model, batch_size, longest_sent, optimizer, label_map, device, save_file=False)

        # break
    return model

def test(padded_X, X_lengths, padded_Y, model, batch_size, longest_sent, optimizer, label_map, device, upos=None, feats=None, fixes=None, results_dir=None, save_file=True):
    y_corr_all = []
    y_pred_all = []
    # test:
    for example_i in range(0, len(padded_X), batch_size):
        # TODO Erase this
        # If last batch size != 16 break
        if example_i + batch_size > len(padded_X):
            break
        X_ids = padded_X[example_i:min(example_i + batch_size, len(padded_X))]
        X_leng = X_lengths[example_i:min(example_i + batch_size, len(X_lengths))]

        Y_ids = padded_Y[example_i:min(example_i + batch_size, len(padded_Y))]

        sorted_data = sorted(zip(X_leng, X_ids, Y_ids), key=lambda pair: pair[0], reverse=True)

        X_leng, X_ids, Y_ids = zip(*sorted_data)
        X_leng, X_ids, Y_ids = list(X_leng), list(X_ids), list(Y_ids)

        # X_leng, X_ids, Y_ids = sorted(zip(X_leng, X_ids, Y_ids))

        # X_ids = torch.tensor(X_ids, dtype=torch.float32)
        # Y_ids = torch.tensor([index for exam in Y_ids for index in exam], dtype=torch.long) - 1

        Y_ids = torch.tensor([index for exam in Y_ids for index in exam], dtype=torch.long) - 1
        Y_ids = Y_ids.to(device)

        X_ids = torch.tensor(X_ids, dtype=torch.float32)
        X_ids = X_ids.to(device)
        if upos:
            upos_ids = torch.tensor(upos_ids, dtype=torch.float32)
            upos_ids = upos_ids.to(device)
        if feats:
            feats_ids = torch.tensor(feats_ids, dtype=torch.float32)
            feats_ids = feats_ids.to(device)
        if fixes:
            fixes_ids = torch.tensor(fixes_ids, dtype=torch.float32)
            fixes_ids = fixes_ids.to(device)


        # Y_ids = torch.tensor(Y_ids, dtype=torch.long) - 1
        # padded_X_ids = torch.tensor(padded_X, dtype=torch.long)
        # padded_Y_ids = torch.tensor([index for exam in padded_Y for index in exam], dtype=torch.long) - 1

        # Forward pass: Compute predicted y by passing x to the model
        # y_pred = model(padded_X, X_lengths)
        # if not all(X_leng):
        #     print('here')
        # y_pred = model(X_ids, X_leng)
        # y_pred = model(new_padded_X_ids, X_lengths)

        with torch.no_grad():
            y_pred = model(X_ids, X_leng)

        y_pred = y_pred.detach().cpu()

        # reshape out_label_ids, create dict for mapping, map all out_label_ids to out_labels, stack them in one array, classification_report(Y_words, out_labels)
        y_pred_reshaped = torch.argmax(y_pred, dim=1)

        y_pred_reshaped = y_pred_reshaped.view(-1, longest_sent).numpy()
        y_corr_reshaped = Y_ids.view(-1, longest_sent).cpu().numpy()

        y_corr = []
        y_pred_tags = []

        for i_y in range(batch_size):
            y_corr_row = []
            y_pred_tags_row = []
            for j_y in range(X_leng[i_y]):
                y_corr_row.append(label_map[y_corr_reshaped[i_y][j_y]])
                y_pred_tags_row.append(label_map[y_pred_reshaped[i_y][j_y]])
            y_corr.append(y_corr_row)
            y_pred_tags.append(y_pred_tags_row)
        # logits = y_pred.detach().cpu().numpy()
        # label_ids = label_ids.to('cpu').numpy()
        # input_mask = input_mask.to('cpu').numpy()

        y_corr_all.extend(y_corr)
        y_pred_all.extend(y_pred_tags)

        # test = torch.argmax(y_pred, dim=1)

        # # TODO ERASE
        # if example_i > 800:
        #     break



        # loss = loss_fn(y_pred, padded_Y_ids)
        # if example_i == 0:
        #     print(epoch, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    report = classification_report(y_corr_all, y_pred_all, digits=4)
    if save_file:
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        # specific_history, overall_accuracy, report = internal_report(y_true, y_pred, digits=4)
        output_eval_file = os.path.join(results_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            writer.write(report)

    print(report)


def preprocess_data(data, tags, fasttext_encoding, longest_sent):
    X_words = []
    Y_words = []
    X_other_words = []

    # for words, labels, other in read_ner:
    for words, labels, other in data:
        X_words.append(words)
        Y_words.append(labels)
        X_other_words.append(other)

    # map sentences to vocab
    # vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
    # fancy nested list comprehension
    X = [[fasttext_encoding[word] for word in sentence] for sentence in X_words]
    # X now looks like:
    # [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]


    # fancy nested list comprehension
    Y = [[tags[tag] for tag in sentence] for sentence in Y_words]
    # Y now looks like:
    # [[1, 2, 3, 3, 3, 1, 4], [5, 5], [4, 5]]


    # num_sent = len(X)

    # get the length of each sentence
    X_lengths = [min(len(sentence), longest_sent) for sentence in X]
    # create an empty matrix with padding tokens
    # pad_token = np.array([0] * 300)
    # longest_sent = max(X_lengths)

    padded_X = []
    # do this to get rid of empty sentences in Y
    new_Y = []
    # copy over the actual sequences
    for i, x_len in enumerate(X_lengths):
        # erase sentences of lenght 0
        if x_len == 0:
            continue
        # a = X[i][:x_len]
        # c = [np.array([0] * 300) for _ in range(longest_sent - x_len)]
        # b = ((longest_sent - x_len) * pad_token)
        new_X = X[i][:x_len] + [np.array([0] * 300) for _ in range(longest_sent - x_len)]
        # cut too long sentences to max sentence size
        padded_X.append(new_X)
        # do this to get rid of empty sentences in Y
        new_Y.append(Y[i])

    # do this to get rid of empty sentences in Y
    Y = new_Y
    num_sent = len(padded_X)
    X_lengths = [x for x in X_lengths if x != 0]
    # get the length of each sentence
    # Y_lengths = [len(sentence) for sentence in Y]
    Y_lengths = [min(len(sentence), longest_sent) for sentence in Y]
    Y_lengths = [x for x in Y_lengths if x != 0]
    # create an empty matrix with padding tokens
    pad_token = tags['<PAD>']
    padded_Y = np.ones((num_sent, longest_sent)) * pad_token
    # copy over the actual sequences
    for i, y_len in enumerate(Y_lengths):
        # erase sentences of lenght 0
        if y_len == 0:
            continue
        sequence = Y[i]
        padded_Y[i, 0:y_len] = sequence[:y_len]

    assert len(padded_X) == padded_Y.shape[0]
    assert len(padded_X) == len(X_lengths)
    assert padded_Y.shape[0] == len(Y_lengths)

    return padded_X, padded_Y, X_lengths, Y_lengths

def run_fastext_LSTM(ner_data_path, device, fasttext_encoding, batch_size, longest_sent, results_dir, cv_part):
    # cv_part = 1
    filename = os.path.join(ner_data_path, "ext_%d_msd.tsv")
    # read_ner = readfile_ner(filename, cv_part)

    num_train_parts = 11
    train_data = []
    for i in range(1, num_train_parts + 1):
        if i != cv_part:
            train_data.extend(readfile_ner(os.path.join(ner_data_path, "ext_%d_msd.tsv"), i))

    test_data = readfile_ner(os.path.join(ner_data_path, "ext_%d_msd.tsv"), cv_part)















    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)








    # padded_X_ids = torch.tensor(padded_X, dtype=torch.long)
    # padded_Y_ids = torch.tensor([index for exam in padded_Y for index in exam], dtype=torch.long) - 1
    tags = {'<PAD>': 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6,
            'I-ORG': 7}
    label_map = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']


    train_X, train_Y, train_X_lengths, train_Y_lengths = preprocess_data(train_data, tags, fasttext_encoding, longest_sent)
    test_X, test_Y, test_X_lengths, test_Y_lengths = preprocess_data(test_data, tags, fasttext_encoding, longest_sent)

    # Construct our model by instantiating the class defined above.
    model = NERLSTM(1, tags, device, longest_sent, batch_size, nb_tags=len(label_map))
    model.to(device)


    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters())

    # a = padded_Y.shape[0]
    # b = len(padded_X)



    # tags = {'<PAD>': 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 2, 'LOC': 2, 'B-PER': 3, 'I-PER': 3, 'PER': 3, 'B-ORG': 4, 'I-ORG': 4, 'ORG': 4}



    model = train(train_X, train_X_lengths, train_Y, test_X, test_X_lengths, test_Y, label_map, model, batch_size, longest_sent, optimizer, loss_fn, device, nb_epoch=100, inside_eval=True)
    test(test_X, test_X_lengths, test_Y, model, batch_size, longest_sent, optimizer, label_map, device, results_dir=results_dir + "/eval_pos_cv_" + str(cv_part), save_file=True)

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    # args.upos = config.getboolean('settings', 'upos')
    # args.feats = config.getboolean('settings', 'feats')
    # args.fixes = config.getboolean('settings', 'fixes')
    model_path = config.get('settings', 'model')
    ner_data_path = config.get('settings', 'ner_data_path')
    results_dir = config.get('settings', 'ner_data_path')
    batch_size = config.getint('settings', 'batch_size')
    longest_sent = config.getint('settings', 'longest_sent')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_gpu = torch.cuda.device_count()

    # Skipgram model :
    # model = fasttext.train_unsupervised('data/data.txt', model='skipgram')

    # or, cbow model :
    # model = fasttext.train_unsupervised(model_path, model='cbow')
    fasttext_encoding = fasttext.load_model(model_path)

    # print(fasttext_encoding.words[:10])

    # vect1 = model.get_word_vector('asdhfasdfgrwarwargfaw937g49dt4w89qsaiugasgihrasg')
    # vect2 = fasttext_encoding['danes']
    # vect3 = model.get_sentence_vector('Katera posoda')

    # filename = filename % cv_part
    # df = pd.read_csv(filename, sep='\t', keep_default_na=False)
    # df = df.fillna('')

    for i in range(1, 12):
        run_fastext_LSTM(ner_data_path, device, fasttext_encoding, batch_size, longest_sent, results_dir, i)
        break
    # run_fastext_LSTM(1)

if __name__ == "__main__":
    main()


########################################################################################################################




# """
# Blog post:
# Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
# https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
# """
#
#
#
#
# # net = NERLSTM()
# # print(net)
#
#
# sent_1_x = ['is', 'it', 'too', 'late', 'now', 'say', 'sorry']
# sent_1_y = ['VB', 'PRP', 'RB', 'RB', 'RB', 'VB', 'JJ']
# sent_2_x = ['ooh', 'ooh']
# sent_2_y = ['NNP', 'NNP']
# sent_3_x = ['sorry', 'yeah']
# sent_3_y = ['JJ', 'NNP']
# X = [sent_1_x, sent_2_x, sent_3_x]
# Y = [sent_1_y, sent_2_y, sent_3_y]
#
# # map sentences to vocab
# vocab = {'<PAD>': 0, 'is': 1, 'it': 2, 'too': 3, 'late': 4, 'now': 5, 'say': 6, 'sorry': 7, 'ooh': 8, 'yeah': 9}
# # fancy nested list comprehension
# X = [[vocab[word] for word in sentence] for sentence in X]
# # X now looks like:
# # [[1, 2, 3, 4, 5, 6, 7], [8, 8], [7, 9]]
#
# tags = {'<PAD>': 0, 'VB': 1, 'PRP': 2, 'RB': 3, 'JJ': 4, 'NNP': 5}
# # fancy nested list comprehension
# Y = [[tags[tag] for tag in sentence] for sentence in Y]
# # Y now looks like:
# # [[1, 2, 3, 3, 3, 1, 4], [5, 5], [4, 5]]
#
# import numpy as np
# # X = [[0, 1, 2, 3, 4, 5, 6],
# #     [7, 7],
# #     [6, 8]]
# # get the length of each sentence
# X_lengths = [len(sentence) for sentence in X]
# # create an empty matrix with padding tokens
# pad_token = vocab['<PAD>']
# longest_sent = max(X_lengths)
# batch_size = len(X)
# padded_X = np.ones((batch_size, longest_sent)) * pad_token
# # copy over the actual sequences
# for i, x_len in enumerate(X_lengths):
#     sequence = X[i]
#     padded_X[i, 0:x_len] = sequence[:x_len]
# # padded_X looks like:
# # array([[ 1.,  2.,  3.,  4.,  5.,  6.,  7.],
# #        [ 8.,  8.,  0.,  0.,  0.,  0.,  0.],
# #        [ 7.,  9.,  0.,  0.,  0.,  0.,  0.]])
#
# # Y = [[1, 2, 3, 3, 3, 1, 4],
# #     [5, 5],
# #     [4, 5]]
# # get the length of each sentence
# Y_lengths = [len(sentence) for sentence in Y]
# # create an empty matrix with padding tokens
# pad_token = tags['<PAD>']
# longest_sent = max(Y_lengths)
# batch_size = len(Y)
# padded_Y = np.ones((batch_size, longest_sent)) * pad_token
# # copy over the actual sequences
# for i, y_len in enumerate(Y_lengths):
#   sequence = Y[i]
#   padded_Y[i, 0:y_len] = sequence[:y_len]
# # padded_Y looks like:
# # array([[ 1.,  2.,  3.,  3.,  3.,  1.,  4.],
# #        [ 5.,  5.,  0.,  0.,  0.,  0.,  0.],
# #        [ 4.,  5.,  0.,  0.,  0.,  0.,  0.]])
#
#
#
#
#
#
#
# # Construct our model by instantiating the class defined above.
# model = NERLSTM(1, fasttext_encoding)
#
# # Construct our loss function and an Optimizer. The call to model.parameters()
# # in the SGD constructor will contain the learnable parameters of the two
# # nn.Linear modules which are members of the model.
# loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
#
# # new_padded_X_ids = [[[token] for token in sent]for sent in padded_X]
# # new_padded_X_ids = torch.tensor(new_padded_X_ids, dtype=torch.long)
# #
# # new_padded_Y_ids = [[[token] for token in sent]for sent in padded_Y]
# # new_padded_Y_ids = torch.tensor(new_padded_Y_ids, dtype=torch.long)
#
# padded_X_ids = torch.tensor(padded_X, dtype=torch.long)
# padded_Y_ids = torch.tensor([index for exam in padded_Y for index in exam], dtype=torch.long) - 1
# # padded_Y_ids = torch.tensor(padded_Y, dtype=torch.long)
# # eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,
# #                            all_prefixes_ids, all_suffixes_ids,
# #                            *all_other_ids)
#
#
#
# # # Run prediction for full data
# # eval_sampler = SequentialSampler(eval_data)
# # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
# # model.eval()
# # eval_loss, eval_accuracy = 0, 0
# # nb_eval_steps, nb_eval_examples = 0, 0
# # y_true = []
# # y_pred = []
# # label_map = {i: label for i, label in enumerate(label_list, 1)}
# # for batch in tqdm(eval_dataloader, desc="Evaluating"):
#
#
# # padded_Y
#
# # Y_ids = torch.zeros(3, 7, 5)
# # TODO UNCOMMENT BELOW AND FIX!!!
# # all_sent = []
# # for li in padded_Y:
# #     sent = []
# #     for el in li:
# #         row = [0, 0, 0, 0, 0]
# #         row[int(el)] = 1
# #         sent.append(row)
# #     all_sent.append(sent)
#
# # final_y = torch.tensor(all_sent, dtype=torch.long)
#
# for t in range(10000):
#     # Forward pass: Compute predicted y by passing x to the model
#     # y_pred = model(padded_X, X_lengths)
#     y_pred = model(padded_X_ids, X_lengths)
#     # y_pred = model(new_padded_X_ids, X_lengths)
#
#     # Compute and print loss
#     # loss = model.loss(y_pred, padded_Y_ids, X_lengths)
#     # loss = model.loss(y_pred, new_padded_Y_ids, X_lengths)
#     # TODO CREATE THIS!!! down
#     criterion = torch.nn.CrossEntropyLoss(size_average=True, ignore_index=-1)
#     # a = [index for exam in padded_Y_ids for index in exam]
#     # loss = criterion(y_pred, [index for exam in padded_Y_ids for index in exam])
#     loss = criterion(y_pred, padded_Y_ids)
#
#     # loss = loss_fn(y_pred, padded_Y_ids)
#     print(t, loss.item())
#
#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
