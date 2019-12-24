#!/bin/env python3
import config as cfg
from src.models.bert_fc import BertFC
from src.utils import AttrDict

from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import TensorDataset, SubsetRandomSampler, DataLoader
from transformers import AdamW

PARAMS = AttrDict({
        'dev_split': 0.2,

        'num_epochs': 1,
        'batch_size': 4,
        'lr': 5e-5,
        'eps': 1e-8,
        'max_grad_norm': 1.0,
        'weight_decay': 0.0,
        'warmup_steps': 0,

        'dev_batch_size': 4,
    })

def fetch_train_dset():
    with open('data/interim/atis.train.pkl', 'rb') as f:
        pkl_dict = pkl.load(f)
        tokens = torch.LongTensor(pkl_dict['tokens'])
        rel_slot_mask = torch.LongTensor(pkl_dict['relevant_slot_mask'])
        attn_mask = torch.LongTensor(pkl_dict['attn_mask'])

        # TODO: Create label mask here
        dset_size = len(pkl_dict['idxes'])
        _idxes = np.zeros((dset_size, cfg.MAX_TOK_LEN), dtype=np.int)
        _idx_mask = np.zeros((dset_size, cfg.MAX_TOK_LEN-1), dtype=np.int)
        for idx, vals, mask in zip(_idxes, pkl_dict['idxes'], _idx_mask):
            idx[:len(vals)] = vals
            mask[:len(vals)-1] = 1
        idxes = torch.LongTensor(_idxes)
        idx_mask = torch.LongTensor(_idx_mask)

        num_intent_classes = len(pkl_dict['intent_dict'].keys())
        num_slot_classes = len(pkl_dict['slot_dict'].keys())
    return (tokens, rel_slot_mask, attn_mask, idxes, idx_mask), (num_intent_classes, num_slot_classes)

# Load dataset
print ("Loading train dataset")
input_tensors, class_sizes = fetch_train_dset()
dset = TensorDataset(*input_tensors)

# Partition into train-dev subsets
dev_size = int(PARAMS.dev_split * len(dset))
indices = np.split(np.random.permutation(len(dset)), [dev_size, len(dset)])
val_idx, train_idx = indices[0], indices[1]
train_sampler, val_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)

# Define dataloaders
train_loader = DataLoader(dset, sampler=train_sampler, batch_size=PARAMS.batch_size)
val_loader = DataLoader(dset, sampler=val_sampler, batch_size=PARAMS.dev_batch_size)

# Init model
print ("\n\nInitializing model, opt")
model = BertFC(num_intent_classes=class_sizes[0], 
               num_slot_classes=class_sizes[1]).to(cfg.device)

# Init optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
      'weight_decay': PARAMS.weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
      'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=PARAMS.lr, eps=PARAMS.eps)

# Initialize loss function
criterion = torch.nn.CrossEntropyLoss()

# Display report, for sanity
print ("\n\n**** Training ****")
print ("Num examples: %d" % (len(dset)))
print ("Num epochs: %d" % (PARAMS.num_epochs))

for epoch in tqdm(range(PARAMS.num_epochs), desc='Epoch'):
    tr_loss, val_loss = 0., 0.

    model.train()
    for idx, batch in enumerate(tqdm(train_loader, leave=False, desc='Train')):
            batch = (t.to(device) for t in batch)

            # TODO: Complete training loop
            # tokens, rel_slot_mask, attn_mask, inp_labels, inp_labels_mask = batch
            # # Get back words labels
            # labels = torch.LongTensor().to(device)
            # for inp_l, inp_l_m in zip(inp_labels, inp_labels_mask):
            #     labels = torch.cat([labels, inp_l[:inp_l_m]], dim=0)
            # logits = model(inp_ids, inp_idx, inp_mask)
            # loss = criterion(logits, labels)
            # tr_loss += loss.item()
            # 
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
