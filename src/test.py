#!/bin/env python3
import config as cfg
import src.utils as utils
from src.models.bert_fc import BertFC
from src.utils import AttrDict

from tqdm import tqdm
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

PARAMS = AttrDict({
    'batch_size': 32,
})

# Load dataset
print ("Loading test dataset")
input_tensors, class_sizes = utils.fetch_test_dset()
dset = TensorDataset(*input_tensors)
test_loader = DataLoader(dset, sampler=SequentialSampler(range(len(dset))), batch_size=PARAMS.batch_size)

print ("Loading model from %s" % (cfg.MODEL_SAVE_FILE))
model = BertFC(num_intent_classes=class_sizes[0], 
               num_slot_classes=class_sizes[1])
model.load_state_dict(torch.load(cfg.MODEL_SAVE_FILE))
model = model.to(cfg.device)

# Display report, for sanity
print ("\n\n**** Inference ****")
print ("Num examples: %d" % (len(dset)))

model.eval()
intent_correct, intent_num, slot_correct, slot_num = 0, 0, 0, 0
for idx, batch in enumerate(tqdm(test_loader, leave=False, desc='Test')):
    batch = (t.to(cfg.device) for t in batch)
    tokens, rel_slot_mask, attn_mask, labels = batch

    bsz = tokens.size(0)
    intent_labels = labels[:, 0].view(-1)
    slot_labels = labels.view(-1)[rel_slot_mask.view(-1) == 1]
    intent_logits, slot_logits = model(tokens, rel_slot_mask, attn_mask)

    intent_preds = torch.argmax(intent_logits, dim=1)
    slot_preds = torch.argmax(slot_logits, dim=1)
    intent_correct += (intent_preds == intent_labels).sum().item()
    intent_num += intent_preds.shape[0]
    slot_correct += (slot_preds == slot_labels).sum().item()
    slot_num += slot_preds.shape[0]

intent_acc = 100. * intent_correct / intent_num
slot_acc = 100. * slot_correct / slot_num
print ('Intent Accuracy: {:.2f} ({}/{}) Slot Accuracy: {:.2f} ({}/{})'
       ''.format(intent_acc, intent_correct, intent_num, 
                 slot_acc, slot_correct, slot_num))
