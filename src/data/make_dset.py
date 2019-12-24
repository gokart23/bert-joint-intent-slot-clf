#!/bin/env python3
import config as cfg
import pickle as pkl
import argparse

from functools import reduce
import itertools

import numpy as np
import torch, torchtext
from torchtext import data, datasets
from transformers import BertTokenizer

cls_token, sep_token = '[CLS]', '[SEP]'

def load_dset(dset_type='train', data_dir='data/processed/'):    
    max_len, sents = 0, []
    print ("DSET TYPE IS %s" % (dset_type))
    with open("%s/atis.sentences.%s.csv" % (data_dir, dset_type), 'r') as f:
        raw = f.read().splitlines()        
        for sent in raw:
            splits = sent.split(',')
            max_len = max(max_len, len(splits))
            # Wordpiece tokenizer (downstream) doesn't play well with the
            # default BertTokenizer's cls_token and sep_token parameters
            splits = [cls_token] + splits[1:-1] + [sep_token]
            sents.append(' '.join(splits))    
    with open("%s/atis.slots.%s.csv" % (data_dir, dset_type), 'r') as f:
        raw = f.read().splitlines()
        slots = [x.split(',') for x in raw]
    with open("%s/atis.intent.%s.csv" % (data_dir, dset_type), 'r') as f:
        intents = f.read().splitlines()
    # Replace BOS slot label with intent
    slots = [[intents[idx]] + x[1:] for idx, x in enumerate(slots)]
    # Lower-case all slot labels - text lower casing handled by Bert tokenizer
    lower_slots = list(map(lambda x: [y.lower() for y in x], slots))
    del intents
    return sents, lower_slots, max_len

def conv_to_idxes(tokenizer, sents, max_tok_len):
    num_sents = len(sents)
    # Choosing pad-token idx to be 0 by default
    tokens = np.zeros((num_sents, max_tok_len), dtype=np.int)
    relevant_tok_mask = np.zeros((num_sents, max_tok_len), dtype=np.int)
    attn_mask = np.zeros((num_sents, max_tok_len), dtype=np.int)
    
    # Use wordpiece tokenizer, and maintain idxes of those tokens that are useful
    # In this case, that is the CLS-token, and the first subword token for every word
    for idx, sent in enumerate(sents):
        sent_toks = tokenizer.tokenize(sent)
        attn_mask[idx, :len(sent_toks)] = 1
        tokens[idx, :len(sent_toks)] = tokenizer.convert_tokens_to_ids(sent_toks)
        relevant_tok_mask[idx, :len(sent_toks)] = [0 
                                                   if (tok.startswith('#')
                                                       or tok in ('EOS')) is True
                                                   else 1
                                                   for idx, tok in enumerate(sent_toks)]
    return tokens, relevant_tok_mask, attn_mask

def to_categorical(labels):
    label_map = {l.lower():i for i, l in enumerate(set(itertools.chain(*labels)))}
    idxes = list()
    for label_list in labels:
        idxes.append([label_map[l.lower()] for l in label_list])
    return label_map, idxes

def main():
    # NOTE: Wordpiece tokenizer doesn't respect 
    # cls_token/sep_token supplied here
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['train', 'test'], default='train',
                        help='Process parallel data and generate interim')
    args = parser.parse_args()

    print ("Processing %s data" % (args.type))
    sents, slots, max_sent_token_len = load_dset(dset_type=args.type)
    # FIXME: This isn't entirely correct, 
    # since max_tok_len is for the *wordpiece* tokenizer len
    assert (max_sent_token_len <= cfg.MAX_TOK_LEN), ("Max naive token len "
                                                "greater than max tok len")
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL_TYPE,
                                              do_basic_tokenize=False,
                                              do_lower_case=True)
    tokens, rel_tok_mask, attn_mask = conv_to_idxes(tokenizer,
                                                    sents, cfg.MAX_TOK_LEN)
    slot_dict, slot_ids = to_categorical(slots)
    fname = "data/interim/atis.%s.pkl" % (args.type)
    with open(fname, 'wb') as f:
        pkl.dump({'tokens': tokens, 'relevant_tok_mask': rel_tok_mask,
                  'attn_mask': attn_mask, 'slot_ids': slot_ids,
                  'slot_dict': slot_dict}, f)
    print ("Written interim data to %s" % (fname))
    print ("Sentences: {:4d}".format(tokens.shape[0]))
    print ("Slot size: {:4d}".format(len(slot_dict.keys())))

if __name__ == "__main__":
    main()
