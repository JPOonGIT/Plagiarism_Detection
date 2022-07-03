from transformers import BertModel

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        kernel = 4
        input_ff = int(755712/(kernel * kernel))
        zwischenlayer1 = int(10000)
        zwischenlayer2 = int(100)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.pooling = nn.MaxPool2d(kernel)
        self.linear11 = nn.Linear(input_ff, zwischenlayer1)
        self.linear12 = nn.Linear(zwischenlayer1, zwischenlayer2)
        self.linear12 = nn.Linear(zwischenlayer2, 1)

    def forward(self, input_data):
        bert_output = self.bert(input_data['input_ids'])
        bert_output = self.pooling(bert_output)
        bert_output = bert_output[0].view(-1, 1)
        x1 = F.relu(self.linear11(bert_output))
        x1 = F.relu(self.linear12(x1))

        return x1

    def train_x1(self, features, labels, loss_function, optimizer, epochs):
        log_interval = 5
        for epoch in range(epochs):
            for index in features:
                for data in features[index]:
                    optimizer.zero_grad()
                    output1 = self.forward(data)
                    loss_x1 = loss_function(output1, labels[index]['multi-author'])
                    loss_x1.backward(retain_graph=False)
                    optimizer.step()
                    if index % log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, index * len(features),
                                                                                       len(features),
                                                                                       100. * index / len(features),
                                                                                       loss_x1.data.item()))

    def test_model(self, features, labels, loss_function):
        log_interval = 5
        for index, data in enumerate(features):
            output = self(data)
            loss = loss_function(output, labels[index]['multi-author'])

            if index % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(index, index * len(data),
                                                                               len(data),
                                                                               100. * index / len(data),
                                                                               loss.data.item()))
