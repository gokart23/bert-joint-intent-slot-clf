import config as cfg

import torch
import pickle as pkl
import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def fetch_dset(filename):
    with open(filename, 'rb') as f:
        pkl_dict = pkl.load(f)
        tokens = torch.LongTensor(pkl_dict['tokens'])
        rel_slot_mask = torch.LongTensor(pkl_dict['relevant_slot_mask'])
        attn_mask = torch.LongTensor(pkl_dict['attn_mask'])

        dset_size = len(pkl_dict['idxes'])
        _idxes = np.zeros((dset_size, cfg.MAX_TOK_LEN), dtype=np.int)
        for idx, vals in zip(_idxes, pkl_dict['idxes']):
            idx[:len(vals)] = vals
        idxes = torch.LongTensor(_idxes)

        num_intent_classes = len(pkl_dict['intent_dict'].keys())
        num_slot_classes = len(pkl_dict['slot_dict'].keys())
    return (tokens, rel_slot_mask, attn_mask, idxes), (num_intent_classes, num_slot_classes)

def fetch_train_dset():
    return fetch_dset('data/interim/atis.train.pkl')

def fetch_test_dset():
    return fetch_dset('data/interim/atis.test.pkl')
