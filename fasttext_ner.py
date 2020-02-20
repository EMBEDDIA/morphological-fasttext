import argparse
import configparser
import csv
import math
import os

import fasttext
import pandas as pd

import torch
import torch.nn as nn
from seqeval.metrics import classification_report
from torch.autograd import Variable

import numpy as np

ud_map = {
    '': 0,
    'ADJ': 1,
    'ADV': 2,
    'INTJ': 3,
    'NOUN': 4,
    'PROPN': 5,
    'VERB': 6,
    'ADP': 7,
    'AUX': 8,
    'CCONJ': 9,
    'DET': 10,
    'NUM': 11,
    'PRON': 12,
    'SCONJ': 13,
    'PUNCT': 14,
    'SYM': 15,
    'X': 16,
    'PART': 17
}

universal_features_map = {
    # lexical features
    'PronType': {
        '': 0,
        'Art': 11,
        'Dem': 1,
        'Emp': 2,
        'Exc': 3,
        'Ind': 4,
        'Int': 5,
        'Neg': 6,
        'Prs': 7,
        'Rcp': 8,
        'Rel': 9,
        'Tot': 10,

    },
    'NumType': {
        '': 0,
        'Card': 7,
        'Dist': 1,
        'Frac': 2,
        'Mult': 3,
        'Ord': 4,
        'Range': 5,
        'Sets': 6
    },
    'Poss': {
        '': 0,
        'Yes': 1
    },
    'Reflex': {
        '': 0,
        'Yes': 1
    },
    'Foreign': {
        '': 0,
        'Yes': 1
    },
    'Abbr': {
        '': 0,
        'Yes': 1
    },

    # Inflectional features (nominal)
    'Gender': {
        '': 0,
        'Com': 4,
        'Fem': 1,
        'Masc': 2,
        'Neut': 3
    },
    'Animacy': {
        '': 0,
        'Anim': 4,
        'Hum': 1,
        'Inan': 2,
        'Nhum': 3,
    },
    'NounClass': {
        '': 0,
        'Bantu1': 20,
        'Bantu2': 1,
        'Bantu3': 2,
        'Bantu4': 3,
        'Bantu5': 4,
        'Bantu6': 5,
        'Bantu7': 6,
        'Bantu8': 7,
        'Bantu9': 8,
        'Bantu10': 9,
        'Bantu11': 10,
        'Bantu12': 11,
        'Bantu13': 12,
        'Bantu14': 13,
        'Bantu15': 14,
        'Bantu16': 15,
        'Bantu17': 16,
        'Bantu18': 17,
        'Bantu19': 18,
        'Bantu20': 19
    },
    'Number': {
        '': 0,
        'Coll': 11,
        'Count': 1,
        'Dual': 2,
        'Grpa': 3,
        'Grpl': 4,
        'Inv': 5,
        'Pauc': 6,
        'Plur': 7,
        'Ptan': 8,
        'Sing': 9,
        'Tri': 10
    },
    'Case': {
        '': 0,
        'Abs': 34,
        'Acc': 1,
        'Erg': 2,
        'Nom': 3,
        'Abe': 4,
        'Ben': 5,
        'Cau': 6,
        'Cmp': 7,
        'Cns': 8,
        'Com': 9,
        'Dat': 10,
        'Dis': 11,
        'Equ': 12,
        'Gen': 13,
        'Ins': 14,
        'Par': 15,
        'Tem': 16,
        'Tra': 17,
        'Voc': 18,
        'Abl': 19,
        'Add': 20,
        'Ade': 21,
        'All': 22,
        'Del': 23,
        'Ela': 24,
        'Ess': 25,
        'Ill': 26,
        'Ine': 27,
        'Lat': 28,
        'Loc': 29,
        'Per': 30,
        'Sub': 31,
        'Sup': 32,
        'Ter': 33
    },
    'Definite': {
        '': 0,
        'Com': 5,
        'Cons': 1,
        'Def': 2,
        'Ind': 3,
        'Spec': 4
    },
    'Degree': {
        'Abs': 0,
        'Cmp': 1,
        'Equ': 2,
        'Pos': 3,
        'Sup': 4
    },

    # Inflectional features (verbal)
    'VerbForm': {
        '': 0,
        'Conv': 8,
        'Fin': 1,
        'Gdv': 2,
        'Ger': 3,
        'Inf': 4,
        'Part': 5,
        'Sup': 6,
        'Vnoun': 7
    },
    'Mood': {
        '': 0,
        'Adm': 12,
        'Cnd': 1,
        'Des': 2,
        'Imp': 3,
        'Ind': 4,
        'Jus': 5,
        'Nec': 6,
        'Opt': 7,
        'Pot': 8,
        'Prp': 9,
        'Qot': 10,
        'Sub': 11
    },
    'Tense': {
        '': 0,
        'Fut': 5,
        'Imp': 1,
        'Past': 2,
        'Pqp': 3,
        'Pres': 4
    },
    'Aspect': {
        '': 0,
        'Hab': 6,
        'Imp': 1,
        'Iter': 2,
        'Perf': 3,
        'Prog': 4,
        'Prosp': 5
    },
    'Voice': {
        '': 0,
        'Act': 8,
        'Antip': 1,
        'Cau': 2,
        'Dir': 3,
        'Inv': 4,
        'Mid': 5,
        'Pass': 6,
        'Rcp': 7
    },
    'Evident': {
        '': 0,
        'Fh': 2,
        'Nfh': 1
    },
    'Polarity': {
        '': 0,
        'Neg': 2,
        'Pos': 1
    },
    'Person': {
        '': 0,
        '0': 5,
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4
    },
    'Polite': {
        '': 0,
        'Elev': 4,
        'Form': 1,
        'Humb': 2,
        'Infm': 3
    },
    'Clusivity': {
        '': 0,
        'Ex': 2,
        'In': 1
    }
}

universal_features_list = universal_features_map.keys()

ud_list = ['', 'ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB', 'ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PRON', 'SCONJ',
           'PUNCT', 'SYM', 'X', 'PART']


prefix_map = {}
suffix_map = {}
prefix_list = []
suffix_list = []

class NERLSTM(nn.Module):
    def __init__(self, nb_lstm_layers, tags, device, sentence_dim, batch_size, nb_tags=7, upos_num=0, prefix_num=0, suffix_num=0, feats_map={}, nb_lstm_units=256, embedding_dim=300, feed_forward_layers=0):
        super(NERLSTM, self).__init__()
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
        self.device = device
        self.on_gpu = True

        self.feed_forward_layers = feed_forward_layers

        # build actual NN
        self.__build_model(upos_num, prefix_num, suffix_num, feats_map)

    def __build_model(self, upos_num=0, prefix_num=0, suffix_num=0, feats_map={}):
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.other_embeddings = nn.ModuleList()
        if upos_num > 0 or feats_map or prefix_num > 1:
            if upos_num > 0:
                self.other_embeddings.append(nn.Embedding(upos_num, 15))

            if feats_map:
                for feat in feats_map:
                    self.other_embeddings.append(nn.Embedding(len(feats_map[feat]), 15))

            if prefix_num > 1:
                self.combined_layer_1 = nn.Linear(self.nb_lstm_units + 15 * len(self.other_embeddings) + 60, self.nb_lstm_units)
            else:
                self.combined_layer_1 = nn.Linear(self.nb_lstm_units + 15 * len(self.other_embeddings),
                                                  self.nb_lstm_units)
            if self.feed_forward_layers == 2:
                self.combined_layer_2 = nn.Linear(self.nb_lstm_units, self.nb_lstm_units)

        elif self.feed_forward_layers == 2:
            self.combined_layer_1 = nn.Linear(self.nb_lstm_units, self.nb_lstm_units)
            self.combined_layer_2 = nn.Linear(self.nb_lstm_units, self.nb_lstm_units)

        else:
            self.combined_layer_1 = nn.Linear(self.nb_lstm_units, self.nb_lstm_units)

            # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

        # create initial values of LSTM
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the weights are of the form (nb_lstm_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if self.on_gpu:
            hidden_a = hidden_a.to(self.device)
            hidden_b = hidden_b.to(self.device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths, upos=None, feats=None, fixes=None):
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True)
        X, _ = self.lstm(X, self.hidden)
        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True, total_length=self.sentence_dim)

        X = X.contiguous()


        if upos is not None or feats is not None or fixes is not None:
            other_embeddings = []
            location_addition = 0

            if upos is not None:
                other_embeddings.append(self.other_embeddings[0](upos))
                location_addition = 1
            if feats is not None:
                for i in range(len(self.other_embeddings) - location_addition):
                    other_embeddings.append(self.other_embeddings[i + location_addition](feats[i]))

            if fixes is not None:
                prefix_embeddings = self.prefix_embeddings(fixes[0])
                suffix_embeddings = self.suffix_embeddings(fixes[1])
                concatenated_tensor = torch.cat((X, prefix_embeddings, suffix_embeddings, *other_embeddings), 2)
            else:
                concatenated_tensor = torch.cat((X, *other_embeddings), 2)
            sequence_output = self.combined_layer_1(concatenated_tensor)
            sequence_output = self.dropout(sequence_output)
            if self.feed_forward_layers == 2:
                sequence_output = self.combined_layer_2(sequence_output)
                sequence_output = self.dropout(sequence_output)

        else:
            sequence_output = self.combined_layer_1(X)
            sequence_output = self.dropout(sequence_output)
            if self.feed_forward_layers == 2:
                sequence_output = self.combined_layer_2(sequence_output)
                sequence_output = self.dropout(sequence_output)

        X = sequence_output.view(-1, sequence_output.shape[2])
        X = self.hidden_to_tag(X)

        return X

    def loss(self, Y_hat, Y, X_lengths):
        Y = Y.view(-1)

        Y_hat = Y_hat.view(-1, self.nb_tags)

        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['<PAD>']
        mask = (Y > tag_pad_token).float()

        nb_tokens = int(torch.sum(mask))

        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y - mask.long()] * mask

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
    first_sentence_i = df['sentence_id'][0]
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
            # if data['word'] == '"':
            #     continue
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
            if not isinstance(data['label'], str) and math.isnan(data['label']):
                labels.append('O')
            else:
                labels.append(data['label'])
        output.append((sentence, labels, others))

    return output

def train(padded_X, X_lengths, padded_Y, test_X, test_X_lengths, test_Y, label_map, model, batch_size, longest_sent, optimizer, criterion, device, nb_epoch=50, upos=None, feats=None, fixes=None, test_upos=None, test_feats=None, test_fixes=None, inside_eval=False):
    for epoch in range(nb_epoch):
        # train
        for example_i in range(0, len(padded_X), batch_size):
            # TODO Erase this
            # If last batch size != 16 break
            if example_i + batch_size > len(padded_X):
                break
            X_ids = padded_X[example_i:min(example_i + batch_size, len(padded_X))]

            upos_ids, feats_ids, fixes_ids = None, None, None

            if upos is not None:
                upos_ids = upos[example_i:min(example_i + batch_size, len(upos))]
            if feats is not None:
                feats_ids = [feat[example_i:min(example_i + batch_size, len(feat))] for feat in feats]
            if fixes is not None:
                fixes_ids = fixes[example_i:min(example_i + batch_size, len(fixes))]
            X_leng = X_lengths[example_i:min(example_i + batch_size, len(X_lengths))]

            Y_ids = padded_Y[example_i:min(example_i + batch_size, len(padded_Y))]



            if upos is not None:
                if feats is not None:
                    if fixes is not None:
                        sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids), key=lambda pair: pair[0],
                                             reverse=True)

                        X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = zip(*sorted_data)
                        X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(
                            upos_ids), list(feats_ids), list(fixes_ids)
                    else:
                        sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, *feats_ids), key=lambda pair: pair[0], reverse=True)

                        X_leng, X_ids, Y_ids, upos_ids, *feats_ids = zip(*sorted_data)
                        X_leng, X_ids, Y_ids, upos_ids = list(X_leng), list(X_ids), list(Y_ids), list(upos_ids)
                        feats_ids = [list(feat_ids) for feat_ids in feats_ids]
                else:
                    sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids), key=lambda pair: pair[0], reverse=True)

                    X_leng, X_ids, Y_ids, upos_ids = zip(*sorted_data)
                    X_leng, X_ids, Y_ids, upos_ids = list(X_leng), list(X_ids), list(Y_ids), list(upos_ids)

            elif feats is not None:
                if fixes is not None:
                    sorted_data = sorted(zip(X_leng, X_ids, Y_ids, feats_ids, fixes_ids),
                                         key=lambda pair: pair[0],
                                         reverse=True)

                    X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = zip(*sorted_data)
                    X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(
                        upos_ids), list(feats_ids), list(fixes_ids)
                else:
                    sorted_data = sorted(zip(X_leng, X_ids, Y_ids, *feats_ids), key=lambda pair: pair[0],
                                         reverse=True)

                    X_leng, X_ids, Y_ids, *feats_ids = zip(*sorted_data)
                    X_leng, X_ids, Y_ids = list(X_leng), list(X_ids), list(Y_ids)
                    feats_ids = [list(feat_ids) for feat_ids in feats_ids]

            elif fixes is not None:
                sorted_data = sorted(zip(X_leng, X_ids, Y_ids, fixes_ids),
                                     key=lambda pair: pair[0],
                                     reverse=True)

                X_leng, X_ids, Y_ids, fixes_ids = zip(*sorted_data)
                X_leng, X_ids, Y_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(fixes_ids)

            else:
                sorted_data = sorted(zip(X_leng, X_ids, Y_ids), key=lambda pair: pair[0], reverse=True)

                X_leng, X_ids, Y_ids = zip(*sorted_data)
                X_leng, X_ids, Y_ids = list(X_leng), list(X_ids), list(Y_ids)

            Y_ids = torch.tensor([index for exam in Y_ids for index in exam], dtype=torch.long) - 1
            Y_ids = Y_ids.to(device)

            X_ids = torch.tensor(X_ids, dtype=torch.float32)
            X_ids = X_ids.to(device)
            if upos is not None:
                upos_ids = torch.tensor(upos_ids, dtype=torch.long)
                upos_ids = upos_ids.to(device)
            if feats is not None:
                for feat_i, feat_ids in enumerate(feats_ids):
                    feat_ids = torch.tensor(feat_ids, dtype=torch.long)
                    feats_ids[feat_i] = feat_ids.to(device)
            if fixes is not None:
                fixes_ids = torch.tensor(fixes_ids, dtype=torch.long)
                fixes_ids = fixes_ids.to(device)

            if fixes is not None:
                y_pred = model(X_ids, X_leng, upos_ids, feats_ids, fixes_ids)
            elif feats is not None:
                y_pred = model(X_ids, X_leng, upos_ids, feats_ids)
            elif upos is not None:
                y_pred = model(X_ids, X_leng, upos_ids)
            else:
                y_pred = model(X_ids, X_leng)

            loss = criterion(y_pred, Y_ids)


            # classification_report(Y_ids, test)
            if example_i == 0:
                print(epoch, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        if inside_eval:
            test(test_X, test_X_lengths, test_Y, model, batch_size, longest_sent, optimizer, label_map, device, upos=test_upos, feats=test_feats, fixes=test_fixes, save_file=False)

    return model

def test(padded_X, X_lengths, padded_Y, model, batch_size, longest_sent, optimizer, label_map, device, upos=None, feats=None, fixes=None, results_dir=None, save_file=True, results_file="eval_results.txt"):
    y_corr_all = []
    y_pred_all = []
    for example_i in range(0, len(padded_X), batch_size):
        # TODO Erase this
        # If last batch size != 16 break
        if example_i + batch_size > len(padded_X):
            break
        X_ids = padded_X[example_i:min(example_i + batch_size, len(padded_X))]
        upos_ids, feats_ids, fixes_ids = None, None, None
        if upos is not None:
            upos_ids = upos[example_i:min(example_i + batch_size, len(upos))]
        if feats is not None:
            feats_ids = [feat[example_i:min(example_i + batch_size, len(feat))] for feat in feats]
        if fixes is not None:
            fixes_ids = fixes[example_i:min(example_i + batch_size, len(fixes))]

        X_leng = X_lengths[example_i:min(example_i + batch_size, len(X_lengths))]

        Y_ids = padded_Y[example_i:min(example_i + batch_size, len(padded_Y))]

        if upos is not None:
            if feats is not None:
                if fixes is not None:
                    sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids),
                                         key=lambda pair: pair[0],
                                         reverse=True)

                    X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = zip(*sorted_data)
                    X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(
                        upos_ids), list(feats_ids), list(fixes_ids)
                else:
                    sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, *feats_ids), key=lambda pair: pair[0],
                                         reverse=True)

                    X_leng, X_ids, Y_ids, upos_ids, *feats_ids = zip(*sorted_data)
                    X_leng, X_ids, Y_ids, upos_ids = list(X_leng), list(X_ids), list(Y_ids), list(upos_ids)
                    feats_ids = [list(feat_ids) for feat_ids in feats_ids]
            else:
                sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids), key=lambda pair: pair[0], reverse=True)

                X_leng, X_ids, Y_ids, upos_ids = zip(*sorted_data)
                X_leng, X_ids, Y_ids, upos_ids = list(X_leng), list(X_ids), list(Y_ids), list(upos_ids)
        elif feats is not None:
            if fixes is not None:
                sorted_data = sorted(zip(X_leng, X_ids, Y_ids, feats_ids, fixes_ids),
                                     key=lambda pair: pair[0],
                                     reverse=True)

                X_leng, X_ids, Y_ids, feats_ids, fixes_ids = zip(*sorted_data)
                X_leng, X_ids, Y_ids, feats_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(feats_ids), list(fixes_ids)
            else:
                sorted_data = sorted(zip(X_leng, X_ids, Y_ids, *feats_ids), key=lambda pair: pair[0],
                                     reverse=True)

                X_leng, X_ids, Y_ids, *feats_ids = zip(*sorted_data)
                X_leng, X_ids, Y_ids, = list(X_leng), list(X_ids), list(Y_ids)
                feats_ids = [list(feat_ids) for feat_ids in feats_ids]
        elif fixes is not None:
            sorted_data = sorted(zip(X_leng, X_ids, Y_ids, upos_ids, feats_ids, fixes_ids),
                                 key=lambda pair: pair[0],
                                 reverse=True)

            X_leng, X_ids, Y_ids, fixes_ids = zip(*sorted_data)
            X_leng, X_ids, Y_ids, fixes_ids = list(X_leng), list(X_ids), list(Y_ids), list(fixes_ids)
        else:
            sorted_data = sorted(zip(X_leng, X_ids, Y_ids), key=lambda pair: pair[0], reverse=True)

            X_leng, X_ids, Y_ids = zip(*sorted_data)
            X_leng, X_ids, Y_ids = list(X_leng), list(X_ids), list(Y_ids)

        Y_ids = torch.tensor([index for exam in Y_ids for index in exam], dtype=torch.long) - 1
        Y_ids = Y_ids.to(device)

        X_ids = torch.tensor(X_ids, dtype=torch.float32)
        X_ids = X_ids.to(device)
        if upos is not None:
            upos_ids = torch.tensor(upos_ids, dtype=torch.long)
            upos_ids = upos_ids.to(device)
        if feats is not None:
            for feat_i, feat_ids in enumerate(feats_ids):
                feat_ids = torch.tensor(feat_ids, dtype=torch.long)
                feats_ids[feat_i] = feat_ids.to(device)
        if fixes is not None:
            fixes_ids = torch.tensor(fixes_ids, dtype=torch.long)
            fixes_ids = fixes_ids.to(device)

        with torch.no_grad():
            if fixes is not None:
                y_pred = model(X_ids, X_leng, upos_ids, feats_ids, fixes_ids)
            elif feats is not None:
                y_pred = model(X_ids, X_leng, upos_ids, feats_ids)
            elif upos is not None:
                y_pred = model(X_ids, X_leng, upos_ids)
            else:
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

        y_corr_all.extend(y_corr)
        y_pred_all.extend(y_pred_tags)

        optimizer.zero_grad()

    report = classification_report(y_corr_all, y_pred_all, digits=4)
    if save_file:
        if not os.path.exists(results_dir):
            os.mkdir(results_dir)
        output_eval_file = os.path.join(results_dir, results_file)

        with open(output_eval_file, "w") as writer:
            writer.write(report)

    print(report)


def preprocess_data(data, tags, fasttext_encoding, longest_sent, upos, feats, fixes):
    X_words = []
    Y_words = []
    X_other_words = []

    # for words, labels, other in read_ner:
    for words, labels, other in data:
        X_words.append(words)
        Y_words.append(labels)
        X_other_words.append(other)

    X = [[fasttext_encoding[word] for word in sentence] for sentence in X_words]

    Y = [[tags[tag] for tag in sentence] for sentence in Y_words]

    if upos:
        X_upos = [[ud_map[other_dict['upos']] for other_dict in sentence] for sentence in X_other_words]
    if feats:
        X_feats = [[] for _ in range(len(universal_features_map))]
        for X_other_sent in X_other_words:
            X_sent_feats = [[] for _ in range(len(universal_features_map))]
            for X_other_word in X_other_sent:
                this_feat_dict = {}
                if X_other_word['feats'] != '_':
                    word_feats = X_other_word['feats'].split('|')
                    for feat in word_feats:
                        feat_split = feat.split('=')
                        this_feat_dict[feat_split[0]] = feat_split[1]

                for index, key in enumerate(universal_features_map):
                    if key in this_feat_dict and this_feat_dict[key] in universal_features_map[key]:
                        X_sent_feats[index].append(universal_features_map[key][this_feat_dict[key]])
                    else:
                        X_sent_feats[index].append(0)

            for ind in range(len(universal_features_map)):
                X_feats[ind].append(X_sent_feats[ind])

    if fixes:
        X_fixes = []
        X_fixes.append([[prefix_map[other_dict['prefix']] for other_dict in sentence] for sentence in X_other_words])
        X_fixes.append([[prefix_map[other_dict['prefix']] for other_dict in sentence] for sentence in X_other_words])

    # get the length of each sentence
    X_lengths = [min(len(sentence), longest_sent) for sentence in X]

    padded_X = []
    if upos:
        new_X_upos = []
    if feats:
        new_X_feats = []
    if fixes:
        new_X_fixes = []
    new_Y = []
    # copy over the actual sequences
    for i, x_len in enumerate(X_lengths):
        # erase sentences of lenght 0
        if x_len == 0:
            continue
        new_X = X[i][:x_len] + [np.array([0] * 300) for _ in range(longest_sent - x_len)]
        padded_X.append(new_X)
        if upos:
            new_X_upos.append(X_upos[i])
        if feats:
            new_X_feats.append([el[i] for el in X_feats])
        if fixes:
            new_X_fixes.append(X_fixes[i])
        # do this to get rid of empty sentences in Y
        new_Y.append(Y[i])

    # do this to get rid of empty sentences in Y
    Y = new_Y
    num_sent = len(padded_X)
    X_lengths = [x for x in X_lengths if x != 0]
    # get the length of each sentence
    Y_lengths = [min(len(sentence), longest_sent) for sentence in Y]
    Y_lengths = [x for x in Y_lengths if x != 0]



    # create an empty matrix with padding tokens
    pad_token = tags['<PAD>']
    padded_Y = np.ones((num_sent, longest_sent)) * pad_token
    if upos:
        padded_X_upos = np.ones((num_sent, longest_sent)) * pad_token
    if feats:
        padded_X_feats = [np.ones((num_sent, longest_sent)) * pad_token for _ in range(len(X_feats))]
    if fixes:
        padded_X_fixes = np.ones((num_sent, longest_sent)) * pad_token

    # copy over the actual sequences
    for i, y_len in enumerate(Y_lengths):
        # erase sentences of lenght 0
        if y_len == 0:
            continue
        if upos:
            padded_X_upos[i, 0:y_len] = new_X_upos[i][:y_len]
        if feats:
            for feat_i in range(len(X_feats)):
                padded_X_feats[feat_i][i, 0:y_len] = new_X_feats[i][feat_i][:y_len]
        if fixes:
            padded_X_fixes[i, 0:y_len] = new_X_fixes[i][:y_len]
        sequence = Y[i]
        padded_Y[i, 0:y_len] = sequence[:y_len]

    assert len(padded_X) == padded_Y.shape[0]
    assert len(padded_X) == len(X_lengths)
    assert padded_Y.shape[0] == len(Y_lengths)

    if not upos:
        padded_X_upos = None
    if not feats:
        padded_X_feats = None
    if not fixes:
        padded_X_fixes = None


    return padded_X, padded_Y, X_lengths, Y_lengths, padded_X_upos, padded_X_feats, padded_X_fixes


def run_fastext_LSTM(ner_data_path, device, fasttext_encoding, batch_size, longest_sent, results_dir, cv_part, upos, feats, fixes, nb_epoch=50, cross_validation=False, feed_forward_layers=0, folds=10, results_file="eval_results.txt"):
    num_train_parts = folds
    # num_train_parts = 10
    train_data = []
    for i in range(1, num_train_parts + 1):
        if i != cv_part:
            train_data.extend(readfile_ner(os.path.join(ner_data_path, "ext_%d_msd.tsv"), i))
            # break

    test_data = readfile_ner(os.path.join(ner_data_path, "ext_%d_msd.tsv"), cv_part)

    tags = {'<PAD>': 0, 'O': 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6,
            'I-ORG': 7}
    label_map = ['O', 'B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG']

    train_X, train_Y, train_X_lengths, train_Y_lengths, train_X_upos, train_X_feats, train_X_fixes = preprocess_data(
        train_data, tags,
        fasttext_encoding,
        longest_sent, upos,
        feats, fixes)
    test_X, test_Y, test_X_lengths, test_Y_lengths, test_X_upos, test_X_feats, test_X_fixes = preprocess_data(test_data,
                                                                                                              tags,
                                                                                                              fasttext_encoding,
                                                                                                              longest_sent,
                                                                                                              upos,
                                                                                                              feats,
                                                                                                              fixes)

    # Construct our model by instantiating the class defined above.

    upos_num = len(ud_map) if upos else 0
    feats_map = universal_features_map if feats else {}

    model = NERLSTM(1, tags, device, longest_sent, batch_size, upos_num=upos_num, feats_map=feats_map, nb_tags=len(label_map), feed_forward_layers=feed_forward_layers)
    model.to(device)


    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters())

    model = train(train_X, train_X_lengths, train_Y, test_X, test_X_lengths, test_Y, label_map, model, batch_size, longest_sent, optimizer, loss_fn, device, nb_epoch=nb_epoch, inside_eval=True, upos=train_X_upos, feats=train_X_feats, fixes=train_X_fixes, test_upos=test_X_upos, test_feats=test_X_feats, test_fixes=test_X_fixes)
    test(test_X, test_X_lengths, test_Y, model, batch_size, longest_sent, optimizer, label_map, device, results_dir=results_dir + "/eval_pos_cv_" + str(cv_part), save_file=True, upos=test_X_upos, feats=test_X_feats, fixes=test_X_fixes, results_file=results_file)

def main():
    global prefix_map
    global suffix_map
    global prefix_list
    global suffix_list
    config = configparser.ConfigParser()
    config.read('config.ini')

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--ner_data_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True,
                        help="Location of fastText .bin for correct language")
    parser.add_argument("--results_file", default=None, type=str, required=True,
                        help="Name of results file")
    args = parser.parse_args()
    model_path = args.model
    ner_data_path = args.ner_data_path
    results_file = args.results_file
    results_dir = config.get('settings', 'ner_data_path')
    batch_size = config.getint('settings', 'batch_size')
    longest_sent = config.getint('settings', 'longest_sent')
    nb_epoch = config.getint('settings', 'nb_epoch')
    upos = config.getboolean('settings', 'upos')
    feats = config.getboolean('settings', 'feats')
    fixes = config.getboolean('settings', 'fixes')
    fixes_path = config.get('settings', 'fixes_path')
    feed_forward_layers = config.getint('settings', 'feed_forward_layers')
    folds = config.getint('settings', 'folds')
    cross_validation = config.getboolean('settings', 'cross_validation')

    if fixes:
        max_prefix_len = 0
        with open(os.path.join(fixes_path, 'prefixes.csv'), 'r') as csvFile:
            reader = csv.reader(csvFile)
            first_el = True
            for row in reader:
                # erase predponsko
                if first_el:
                    first_el = False
                    continue
                # erase prefixes with length smaller than 2 characters
                if len(row[0]) > 2:
                    # add new prefix and erase _
                    prefix_list.append(row[0][:-1])

                # find longest prefix
                if len(row[0]) - 1 > max_prefix_len:
                    max_prefix_len = len(row[0]) - 1

        max_suffix_len = 0
        with open(os.path.join(fixes_path, 'suffixes.csv'), 'r') as csvFile:
            reader = csv.reader(csvFile)
            first_el = True
            for row in reader:
                # erase priponsko
                if first_el:
                    first_el = False
                    continue

                # erase suffixes with length smaller than 2 characters
                if len(row[0]) > 2:
                    # add new suffix and erase _
                    suffix_list.append(row[0][1:])

                if len(row[0]) - 1 > max_suffix_len:
                    max_suffix_len = len(row[0]) - 1

        suffix_list = [''] + suffix_list

        prefix_list = [''] + prefix_list

    prefix_map = {val: i for i, val in enumerate(prefix_list)}
    suffix_map = {val: i for i, val in enumerate(suffix_list)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    fasttext_encoding = fasttext.load_model(model_path)

    for i in range(1, folds + 1):
        run_fastext_LSTM(ner_data_path, device, fasttext_encoding, batch_size, longest_sent, results_dir, i, upos, feats, fixes, nb_epoch=nb_epoch, cross_validation=cross_validation, feed_forward_layers=feed_forward_layers, folds=folds, results_file=results_file)
        if not cross_validation:
            break

if __name__ == "__main__":
    main()
