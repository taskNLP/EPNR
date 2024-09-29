import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
import time
from utils import decode
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    None_Relation = '<None>'
    Relation = '<R>'

    def __init__(self):
        self.label2id = {self.None_Relation: 0, self.Relation: 1}
        self.id2label = {0: self.None_Relation, 1: self.Relation}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]

def collate_fn(data):
    bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length, entity_list, relation_list = map(list, zip(*data))


    sent_length = torch.LongTensor(sent_length)
    bert_inputs = pad_sequence(bert_inputs, True)
    entity_length = torch.LongTensor(entity_length)
    batch_size = bert_inputs.size(0)
    max_tok = np.max([x.shape[0] for x in pieces2word])
    max_pie = np.max([x.shape[1] for x in pieces2word])
    max_entity = np.max([x.shape[0] for x in pieces2entity])

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0]] = x
        return new_data

    def fill2d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    labels_mat = torch.zeros((batch_size, max_entity, max_entity), dtype=torch.long)
    grid_labels = fill2d(grid_labels, labels_mat)


    grid_mask2d_mat = torch.zeros((batch_size, max_entity, max_entity), dtype=torch.bool)
    grid_mask2d = fill2d(grid_mask2d, grid_mask2d_mat)

    entity_mask_mat = torch.ones((batch_size, max_entity, max_entity), dtype=torch.long)
    entity_mask = fill2d(entity_mask, entity_mask_mat)

    pieces2word_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.long)
    pieces2word = fill2d(pieces2word, pieces2word_mat)

    pieces2entity_mat = torch.zeros((batch_size, max_entity, max_tok), dtype=torch.long)
    pieces2entity = fill2d(pieces2entity, pieces2entity_mat)



    return bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length, entity_list, relation_list


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length, entity_list, relation_list):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.entity_mask = entity_mask
        self.pieces2word = pieces2word
        self.pieces2entity = pieces2entity
        self.sent_length = sent_length
        self.entity_length = entity_length
        self.entity_list = entity_list
        self.relation_list = relation_list


    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
               torch.LongTensor(self.grid_labels[item]), \
               torch.LongTensor(self.grid_mask2d[item]), \
               torch.LongTensor(self.entity_mask[item]), \
               torch.LongTensor(self.pieces2word[item]), \
               torch.LongTensor(self.pieces2entity[item]), \
               self.sent_length[item], \
               self.entity_length[item], \
               self.entity_list[item], \
               self.relation_list[item]


    def __len__(self):
        return len(self.bert_inputs)

def relation2label(relation, entity_list, grid_labels, vocab):
    if relation["subject_type"] == "relation":
        head = relation2label(relation["subject_index"], entity_list, grid_labels, vocab)
    else:
        head = entity_list.index(relation["subject_index"])
    if relation["object_type"] == "relation":
        tail = relation2label(relation["object_index"], entity_list, grid_labels, vocab)
    else:
        tail = entity_list.index(relation["object_index"])
    grid_labels[head, tail] = vocab.label_to_id(relation["relation"])
    return head

def relation2entitylist(relation, entity_list):
    if relation["subject_type"] == "relation":
        head = relation2entitylist(relation["subject_index"], entity_list)
    else:
        head = [entity_list.index(relation["subject_index"])]
    if relation["object_type"] == "relation":
        tail = relation2entitylist(relation["object_index"], entity_list)
    else:
        tail = [entity_list.index(relation["object_index"])]

    return head + tail

def relation2list(relation, entity_list, vocab):
    if relation["subject_type"] == "relation":
        head = relation2list(relation["subject_index"], entity_list, vocab)
    else:
        head = entity_list.index(relation["subject_index"])
    if relation["object_type"] == "relation":
        tail = relation2list(relation["object_index"], entity_list, vocab)
    else:
        tail = entity_list.index(relation["object_index"])

    return [head, tail, vocab.label_to_id(relation["relation"])]

def process_bert(data, tokenizer, vocab):

    bert_inputs = []   
    grid_labels = []
    grid_mask2d = []
    pieces2word = [] 
    pieces2entity = []
    entity_list = []
    entity_mask = []
    relation_list = []
    entity_length = []
    sent_length = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(pieces)
        _bert_inputs = np.array([tokenizer.cls_token_id] +_bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        entity_num = len(instance["ner"])
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=bool)
        _pieces2entity = np.zeros((entity_num, length), dtype=bool)
        _grid_mask2d = np.ones((entity_num, entity_num), dtype=bool)
        _entity_mask = np.zeros((entity_num, entity_num), dtype=int)
        _grid_labels = np.zeros((entity_num, entity_num), dtype=int)
        _entity_list = [indexs["index"] for indexs in instance["ner"]]


        for k in range(entity_num):
            _entity_mask[k, :] += k
            _entity_mask[:, k] -= k

        for i in range(entity_num):
            for j in range(entity_num):
                if _entity_mask[i, j] < 0:
                    _entity_mask[i, j] = dis2idx[-_entity_mask[i, j]] + 9
                else:
                    _entity_mask[i, j] = dis2idx[_entity_mask[i, j]]
        _entity_mask[_entity_mask == 0] = 19

        if tokenizer is not None:
            for i, entity in enumerate(_entity_list):
                for j in entity:
                    _pieces2entity[i, j] = 1

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        _reltaion_list = []
        for relation in instance["relations"]:
            entity_index_list = relation2entitylist(relation, _entity_list)
            entity_num = len(entity_index_list)
            for i in range(entity_num-1):
                for j in range(i+1, entity_num):
                    _grid_labels[entity_index_list[i], entity_index_list[j]] = 1
            relation2label(relation, _entity_list, _grid_labels, vocab)
            _reltaion_list.append(str(relation2list(relation, _entity_list, vocab)))

        sent_length.append(length)
        relation_list.append(_reltaion_list)
        entity_length.append(len(instance["ner"]))
        bert_inputs.append(_bert_inputs)
        grid_labels.append(_grid_labels)
        grid_mask2d.append(_grid_mask2d)
        entity_mask.append(_entity_mask)
        pieces2word.append(_pieces2word)
        pieces2entity.append(_pieces2entity)
        entity_list.append(_entity_list)

    return bert_inputs, grid_labels, grid_mask2d, entity_mask, pieces2word, pieces2entity, sent_length, entity_length, entity_list, relation_list

def fill_vocab_by_data(vocab, data):
    vocab.add_label(data["relation"].replace(" ", "_"))
    if data["subject_type"] == "relation":
        fill_vocab_by_data(vocab, data["subject_index"])
    try:
        if data["object_type"] == "relation":
            fill_vocab_by_data(vocab, data["object_index"])
    except:
        print(data)
        exit()

def fill_vocab(vocab, dataset):
    for instance in dataset:
        for relation in instance["relations"]:
            fill_vocab_by_data(vocab, relation)

def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data1 = json.load(f)
        train_data = []
        for data in train_data1:
            if data["ner"]:
                train_data.append(data)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data1 = json.load(f)
        dev_data = []
        for data in dev_data1:
            if data["ner"]:
                dev_data.append(data)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data1 = json.load(f)
        test_data = []
        for data in test_data1:
            if data["ner"]:
                test_data.append(data)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    fill_vocab(vocab, train_data)
    fill_vocab(vocab, dev_data)
    fill_vocab(vocab, test_data)
    print("class_label = ", len(vocab.label2id))

    table = pt.PrettyTable([config.dataset, 'sentences'])
    table.add_row(['train', len(train_data)])
    table.add_row(['dev', len(dev_data)])
    table.add_row(['test', len(test_data)])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    print("train\n")
    train_dataset = RelationDataset(*process_bert(train_data, tokenizer, vocab))
    print("dev_dataset\n")
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    print("test_dataset\n")
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)

