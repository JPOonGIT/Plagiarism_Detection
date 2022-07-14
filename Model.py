from transformers import BertModel

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        zwischenlayer = 100
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.linear11 = nn.Linear(768, zwischenlayer)
        #self.linear21 = nn.Linear(768, zwischenlayer)
        #.linear31 = nn.Linear(768, zwischenlayer)
        #self.linear41 = nn.Linear(768, zwischenlayer)
        self.linear12 = nn.Linear(zwischenlayer, 1)
        #self.linear22 = nn.Linear(zwischenlayer, 1)
        #self.linear32 = nn.Linear(zwischenlayer, 1)
        #self.linear42 = nn.Linear(zwischenlayer, 1)

    def forward(self, input_ids, attention_mask = None):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        bert_output = bert_output[0].view(-1,1)
        x1 = F.relu(self.linear11(bert_output))
        x1 = F.relu(self.linear12(x1))
        x2 = F.relu(self.linear21(bert_output))
        x2 = F.relu(self.linear22(x2))
        '''x3 = F.relu(self.linear31(bert_output))
        x3 = F.relu(self.linear32(x3))
        x4 = F.relu(self.linear41(bert_output))
        x4 = F.relu(self.linear42(x4))'''

        return x1 ,x2 #, x3, x4

    def train_x1(self,dataframe, loss_function, optimizer, epochs):
        log_interval = 5
        for epoch in range(epochs):
            for index, data in enumerate(dataframe):
                data, target = data[0], data[1]
                optimizer.zero_grad()
                output1 = self(data)[0]
                loss_x1 = loss_function(output1, target)
                loss_x1.backward(retain_graph=False)
                optimizer.step()
                if index % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, index * len(data),
                                                                                   len(data),
                                                                                   100. * index / len(data),
                                                                                   loss_x1.data.item()))

    def train_x2(self, dataframe, loss_function, optimizer, epochs):
        log_interval = 5
        for epoch in range(epochs):
            for index, data in enumerate(dataframe):
                data, target = data[0], data[1]
                optimizer.zero_grad()
                output2 = self(data)[1]
                loss_x2 = loss_function(output2, target)
                loss_x2.backward(retain_graph=False)
                optimizer.step()
                if index % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs,
                                                                                   index * len(data),
                                                                                   len(data),
                                                                                   100. * index / len(
                                                                                       data),
                                                                                   loss_x2.data.item()))

    def train_x3(self, dataframe, loss_function, optimizer, epochs):
        log_interval = 5
        for epoch in range(epochs):
            for index, data in enumerate(dataframe):
                data, target = data[0], data[1]
                optimizer.zero_grad()
                output3 = self(data)[2]
                loss_x3 = loss_function(output3, target)
                loss_x3.backward(retain_graph=False)
                optimizer.step()
                if index % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, index * len(data),
                                                                                   len(data),
                                                                                   100. * index / len(data),
                                                                                   loss_x3.data.item()))

    def train_x4(self, dataframe, loss_function, optimizer, epochs):
        log_interval = 5
        for epoch in range(epochs):
            for index, data in enumerate(dataframe):
                data, target = data[0], data[1]
                optimizer.zero_grad()
                output4 = self(data)[3]
                loss_x4 = loss_function(output4, target)
                loss_x4.backward(retain_graph=False)
                optimizer.step()
                if index % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epochs, index * len(data),
                                                                                   len(data),
                                                                                   100. * index / len(data),
                                                                                   loss_x4.data.item()))



    def test_model(self,dataframe, loss_function):
            log_interval = 1
            for index, data in enumerate(dataframe):
                data, target = data[0], data[1]
                output = self(data)
                loss = loss_function(output, target)

                if index % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(index, index * len(data),
                                                                                   len(data),
                                                                                   100. * index / len(data),
                                                                                   loss.data.item()))
