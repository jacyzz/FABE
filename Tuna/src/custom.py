import copy
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List, Union, Any
import json
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

IGNORE_INDEX = -100


class TunaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = kwargs["args"].margin
        self.mle_weight = kwargs["args"].mle_weight

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        bs, num_cand, seq_len = inputs["input_ids"].size()
        input_ids = inputs["input_ids"].view(bs * num_cand, seq_len)
        attention_mask = inputs["attention_mask"].view(bs * num_cand, seq_len)
        labels = inputs["labels"]
        label_mask = labels.ne(IGNORE_INDEX)

        # model.to(input_ids.device)  #######################
        # output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels.view(bs * num_cand, seq_len), return_dict=True)
        output = model(
            input_ids=input_ids,
            labels=labels.view(bs * num_cand, seq_len),
            return_dict=True,
        )
        # print(output)
        # print(output)
        logits = output.logits.view(bs, num_cand, seq_len, -1)
        lprobs = F.log_softmax(logits, dim=-1)
        labels.masked_fill_(~label_mask, 0)
        lprobs = (
            lprobs[:, :, :-1, :]
            .gather(dim=-1, index=labels[:, :, 1:, None])
            .squeeze(-1)
        )  # bs, num_cand, seq_len-1
        token_lprobs = (lprobs * label_mask[:, :, 1:].type_as(lprobs)).sum(
            dim=-1
        )  # bs, num_cand
        
        # Vectorized implementation of margin ranking loss
        # This ensures the computation graph is not broken
        ranking_loss = 0
        for i in range(1, num_cand):
            pos_scores = token_lprobs[:, :-i]
            neg_scores = token_lprobs[:, i:]
            
            # The core formula for margin ranking loss is max(0, - (pos_scores - neg_scores) + margin)
            # We use F.relu which is equivalent to max(0, x)
            margin = self.margin * i
            loss_per_pair = F.relu(neg_scores - pos_scores + margin)
            ranking_loss += loss_per_pair.mean()

        # Normalize the ranking loss by the number of comparisons
        if num_cand > 1:
            ranking_loss = ranking_loss / (num_cand - 1)

        final_loss = self.mle_weight * output.loss + ranking_loss

        return final_loss
