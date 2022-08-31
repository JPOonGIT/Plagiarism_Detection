from transformers import BertModel
import Data_Preprocessing
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        kernel = 4
        input_ff = int(393216/(kernel * kernel))
        zwischenlayer1 = int(100)
        zwischenlayer2 = int(10)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.pooling = nn.MaxPool2d(kernel)
        self.linear11 = nn.Linear(input_ff, zwischenlayer1)
        self.linear12 = nn.Linear(zwischenlayer1, zwischenlayer2)
        self.linear13 = nn.Linear(zwischenlayer2, 2)

        self.linear21 = nn.Linear(input_ff, zwischenlayer1)
        self.linear22 = nn.Linear(zwischenlayer1, zwischenlayer2)
        self.linear23 = nn.Linear(zwischenlayer2, 2)

    def forward_x1(self, input_data):
        bert_output = self.bert(**input_data)
        bert_output = self.pooling(bert_output[0])
        bert_output = bert_output[0].view(-1, 24576)
        x1 = torch.sigmoid(self.linear11(bert_output))
        x1 = torch.sigmoid(self.linear12(x1))
        x1 = torch.sigmoid(self.linear13(x1))

        return x1

    def forward_x2(self, input_data):
        bert_output = self.bert(**input_data)
        bert_output = self.pooling(bert_output[0])
        bert_output = bert_output[0].view(-1, 24576)
        x1 = torch.sigmoid(self.linear21(bert_output))
        x1 = torch.sigmoid(self.linear22(x1))
        x1 = torch.sigmoid(self.linear23(x1))

        return x1

    def train_x1(self, features, labels, loss_function, optimizer, epochs):
        log_interval = 5
        evaluation_list = list()
        for epoch in range(epochs):
            for i, data in enumerate(features):
                data = Data_Preprocessing.preprocessing(data)
                optimizer.zero_grad()
                output1 = self.forward_x1(data)
                target =  torch.tensor(labels[i])
                target = target.to(torch.float32)
                # print(torch.tensor(output1[0][0]),torch.tensor(float(labels[i]['multi-author'])))
                print(output1[0],target)
                # loss_x1 = loss_function(torch.tensor(output1[0][0]), torch.tensor(float(labels[i]['multi-author'])))
                loss_x1 = loss_function(output1[0],target)
                evaluation_list.append(([output1,torch.tensor(labels[i]), loss_x1]))
                print(i, '  ist=', output1, ' soll=', torch.tensor(labels[i]), ' Loss=', loss_x1)
                loss_x1.backward()
                optimizer.step()

        return evaluation_list

    def train_x2(self, features, labels, loss_function, optimizer, epochs):
        log_interval = 5
        evaluation_list = list()
        for epoch in range(epochs):
            for i, data in enumerate(features):
                data = Data_Preprocessing.preprocessing(data)
                optimizer.zero_grad()
                output1 = self.forward_x2(data)
                target = torch.tensor(labels[i])
                target = target.to(torch.float32)
                # print(torch.tensor(output1[0][0]),torch.tensor(float(labels[i]['multi-author'])))
                print(output1, target)
                # loss_x1 = loss_function(torch.tensor(output1[0][0]), torch.tensor(float(labels[i]['multi-author'])))
                loss_x1 = loss_function(output1[0], target)
                evaluation_list.append(([output1, torch.tensor(labels[i]), loss_x1]))
                print(i, '  ist=', output1, ' soll=', torch.tensor(labels[i]), ' Loss=', loss_x1)
                loss_x1.backward()
                optimizer.step()

        return evaluation_list

    def test_model_x1(self, features, labels, loss_function):
        log_interval = 5
        evaluation_list = list()

        for index, data in enumerate(features):
            data = Data_Preprocessing.preprocessing(data)
            output = self.forward_x1(data)
            target = torch.tensor(labels[index])
            target = target.to(torch.float32)
            loss = loss_function(output[0], target)
            evaluation_list.append(([output,torch.tensor(labels[index]['multi-author']), loss]))

        return evaluation_list

    def test_model_x2(self, features, labels, loss_function):
        log_interval = 5
        evaluation_list = list()

        for index, data in enumerate(features):
            data = Data_Preprocessing.preprocessing(data)
            output = self.forward_x2(data)
            target = torch.tensor(labels[index])
            target = target.to(torch.float32)
            loss = loss_function(output[0], target)
            evaluation_list.append(([output, torch.tensor(labels[index]), loss]))

        return evaluation_list