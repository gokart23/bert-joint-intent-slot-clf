#!/bin/env python3
import config as cfg
import src.utils as utils
import src.data.make_dset as dset
from src.models.bert_fc import BertFC

import torch
import pickle as pkl
import sys, os, argparse
from collections import defaultdict
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=cfg.MODEL_SAVE_FILE)
parser.add_argument('--test-dset', type=str, 
                    default='data/interim/atis.test.pkl')

args = parser.parse_args()
if not os.path.isfile(args.model):
    print ("Invalid model filepath %s - please provide valid path"
           % (args.model))

with open(args.test_dset, 'rb') as f:
    pkl_dict = pkl.load(f)
    intent_dict, slot_dict = pkl_dict['intent_dict'], pkl_dict['slot_dict']
    rev_intent, rev_slot = {v:k for k,v in intent_dict.items()}, {v:k for k,v in slot_dict.items()}

num_intent_classes, num_slot_classes = len(intent_dict.keys()), len(slot_dict.keys())
model = BertFC(num_intent_classes=num_intent_classes,
               num_slot_classes=num_slot_classes)
print ("Attempting to load trained model with %d intent classes"
       " and %d slot classes" % (num_intent_classes, num_slot_classes))
model.load_state_dict(torch.load(args.model))
model = model.to(cfg.device)
model.eval()

print ("Model loaded successfully, max token length is %d" % (cfg.MAX_TOK_LEN))

tokenizer = BertTokenizer.from_pretrained(cfg.MODEL_TYPE,
                                          do_basic_tokenize=False,
                                          do_lower_case=True)
# REPL loop
while True:
    # Read
    sys.stdout.write("\n> ")
    orig_sentence = input()
    sentence = dset.cls_token + ' ' + orig_sentence + ' ' + dset.sep_token

    # Eval (inference)
    inp = dset.conv_to_idxes(tokenizer, [sentence], cfg.MAX_TOK_LEN)
    tokens, rel_slot_mask, attn_mask = (torch.LongTensor(t).to(cfg.device) for t in inp)
    logits = model(tokens, rel_slot_mask, attn_mask)
    intent_preds, slot_preds = (torch.argmax(t, dim=1).tolist() for t in logits)

    # Pretty print
    print ("Intent: ", rev_intent[intent_preds[0]])
    print ("Slots: ", ' '.join(["%s(%s)" % (w, rev_slot[x].upper())
            for w, x in zip(orig_sentence.split(), slot_preds)]))
    pred_dict = defaultdict(lambda: "")
    for w, x in zip(orig_sentence.split(), slot_preds):
        slot_class = rev_slot[x].upper().split('-')
        if len(slot_class) > 1: # not an O class
            pred_dict[slot_class[1]] += ' ' + w
    print ("--------------")
    print ("\n".join(["%s: %s" % (k, v) for k,v in pred_dict.items()]))
