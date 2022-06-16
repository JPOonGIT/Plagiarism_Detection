from transformers import BertModel
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bert = BertModel.BertModel.from_pretrained("bert-base-uncased")
        self.linear11 = nn.Linear()
        self.linear21 = nn.Linear()
        self.linear31 = nn.Linear()
        self.linear41 = nn.Linear()
        self.linear12 = nn.Linear()
        self.linear22 = nn.Linear()
        self.linear32 = nn.Linear()
        self.linear42 = nn.Linear()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_output = bert_output[:, 0, :].view(-1, 768)
        x1 = F.relu(self.linear11(bert_output))
        x1 = F.relu(self.linear12(x1))
        x2 = F.relu(self.linear21(bert_output))
        x2 = F.relu(self.linear22(x2))
        x3 = F.relu(self.linear31(bert_output))
        x3 = F.relu(self.linear32(x3))
        x4 = F.relu(self.linear41(bert_output))
        x4 = F.relu(self.linear42(x4))

        return x1, x2, x3, x4
