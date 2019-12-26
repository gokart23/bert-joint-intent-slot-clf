#!/bin/env python3
import config as cfg
import src.utils as utils
from src.models.bert_fc import BertFC
from src.utils import AttrDict

from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import TensorDataset, SubsetRandomSampler, DataLoader
from transformers import AdamW

PARAMS = AttrDict({
        'num_epochs': 30,
        'batch_size': 32,
        'lr': 5e-5,
        'eps': 1e-8,
        'max_grad_norm': 1.0,
        'weight_decay': 0.0,
        'warmup_steps': 0,

        'dev_split': 0.2,
        'dev_batch_size': 32,
    })

# Load dataset
print ("Loading train dataset")
input_tensors, class_sizes = utils.fetch_train_dset()
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
intent_loss = torch.nn.CrossEntropyLoss()
slot_loss = torch.nn.CrossEntropyLoss()

# Display report, for sanity
print ("\n\n**** Training ****")
print ("Num examples: %d" % (len(dset)))
print ("Num epochs: %d" % (PARAMS.num_epochs))

for epoch in tqdm(range(PARAMS.num_epochs), desc='Epoch'):
    tr_loss, val_loss = 0., 0.
    model.train()
    for idx, batch in enumerate(tqdm(train_loader, leave=False)):
        batch = (t.to(cfg.device) for t in batch)
        tokens, rel_slot_mask, attn_mask, labels = batch

        bsz = tokens.size(0)
        intent_labels = labels[:, 0].view(-1)
        slot_labels = labels.view(-1)[rel_slot_mask.view(-1) == 1]
        assert (intent_labels.size(0) == bsz), "Intent labels don't match"
        assert (slot_labels.size(0) == rel_slot_mask.sum()), "Slot labels don't match"

        intent_logits, slot_logits = model(tokens, rel_slot_mask, attn_mask)
        i_loss = intent_loss(intent_logits, intent_labels)
        s_loss = slot_loss(slot_logits, slot_labels)
        loss = i_loss + s_loss

        tr_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    intent_correct, intent_num, slot_correct, slot_num = 0, 0, 0, 0
    for idx, batch in enumerate(tqdm(val_loader, leave=False, desc='Eval')):
        batch = (t.to(cfg.device) for t in batch)
        tokens, rel_slot_mask, attn_mask, labels = batch

        bsz = tokens.size(0)
        intent_labels = labels[:, 0].view(-1)
        slot_labels = labels.view(-1)[rel_slot_mask.view(-1) == 1]
        intent_logits, slot_logits = model(tokens, rel_slot_mask, attn_mask)

        i_loss = intent_loss(intent_logits, intent_labels)
        s_loss = slot_loss(slot_logits, slot_labels)
        loss = i_loss + s_loss
        val_loss += loss.item()

        intent_preds = torch.argmax(intent_logits, dim=1)
        slot_preds = torch.argmax(slot_logits, dim=1)
        intent_correct += (intent_preds == intent_labels).sum().item()
        intent_num += intent_preds.shape[0]
        slot_correct += (slot_preds == slot_labels).sum().item()
        slot_num += slot_preds.shape[0]

    intent_acc = 100. * intent_correct / intent_num
    slot_acc = 100. * slot_correct / slot_num
    tqdm.write ('Train loss: {:.4f}, Val loss: {:.4f},'
           'Intent Accuracy: {:.2f} ({}/{}) Slot Accuracy: {:.2f} ({}/{})'
           ''.format(tr_loss, val_loss, intent_acc, intent_correct, intent_num,
                     slot_acc, slot_correct, slot_num))
torch.save(model.state_dict(), cfg.MODEL_SAVE_FILE)
print ("Saved model to %s" % (cfg.MODEL_SAVE_FILE))
