import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class LanguageEncoder(nn.Module):
    def __init__(self, out_dim=256, pretrained_model="bert-base-uncased"):
        super(LanguageEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        bert_hidden = self.bert.config.hidden_size
        self.proj = nn.Linear(bert_hidden, out_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # Use [CLS] token embedding from BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        cls_token = outputs.last_hidden_state[:, 0, :]  # CLS token
        x = self.proj(cls_token)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x
