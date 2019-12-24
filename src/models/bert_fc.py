import config as cfg

import torch
import torch.nn as nn
from transformers import BertModel

class BertFC(nn.Module):
    def __init__(self, num_intent_classes, num_slot_classes, type=cfg.MODEL_TYPE):
        super(BertFC, self).__init__()
        self.bert = BertModel.from_pretrained(type)
        self.bert_hidden_dim = self.bert.embeddings.word_embeddings.embedding_dim

        self.drop = nn.Dropout()
        self.intent_fc = nn.Linear(self.bert_hidden_dim, num_intent_classes)
        self.slot_fc = nn.Linear(self.bert_hidden_dim, num_slot_classes)

    def forward(self, inp, relevant_slot_mask, attn_mask):
        # bsz x max_tok_len -> bsz x max_tok_len x bert_hidden_dim
        encoded_states, _ = self.bert(inp, attention_mask=attn_mask,
                                      output_all_encoded_layers=False)
        joint_enc = self.drop(encoded_states)

        # intents: bsz x max_tok_len x bert_hidden_dim -> bsz x bert_hidden_dim
        intents = joint_enc[:, 0, :]

        # bsz x max_tok_len x bert_hidden_dim -> bsz * max_tok_len x bert_hidden_dim
        x = joint_enc.view(-1, self.bert_hidden_dim)
        # choose slots which actually matter
        slots = x[relevant_slot_mask.view(-1) == 1, :]

        # bsz x bert_hidden_dim -> bsz x num_intent_classes
        intent_logits = self.intent_fc(intents)
        # * x bert_hidden_dim -> * x num_slot_classes
        slot_logits = self.slot_fc(slots)

        return intent_logits, slot_logits
