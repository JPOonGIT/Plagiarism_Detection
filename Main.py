import Data_Preprocessing
import Model
import Doc_import
import torch
import torch.nn as nn

fname_train_labels = '/Users/alexbeetz/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl'
fname_train_features = '/Users/alexbeetz/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl'
fname_test_labels = '/Users/alexbeetz/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small-truth.jsonl'
fname_test_features = '/Users/alexbeetz/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl'
pfad_pan21 = r"C:\Users\jan-p\Documents\Uni Wuppertal\Semester 2\Information Retrival\Projekt\Data\pan21"

train_evaluation = list()
test_evaluation = list()

train_labels_task1 = list()
train_labels_task2 = list()

train_features_task1 = list()
train_features_task2 = list()

test_labels_task1 = list()
test_labels_task2 = list()

test_features_task1 = list()
test_features_task2 = list()

'''
train_features_task1, train_labels_task1, test_features_task1, test_labels_task1 = Doc_import.import_pan21(pfad_pan21, returns= 10)
'''
train_labels_task1, train_features_task1 = Doc_import.text_preprocessing(fname_train_labels, fname_train_features)

train_labels_task2, train_features_task2 = Doc_import.text_preprocessing(fname_train_labels, fname_train_features)

#test_labels_task2, test_features_task2 = Doc_import.text_preprocessing(fname_test_labels, fname_test_features)



train_labels_task1_bin = Data_Preprocessing.preprocessing_task2_binary(train_labels_task1)
train_labels_task2_bin = Data_Preprocessing.preprocessing_task2_binary(train_labels_task2)

#print((train_labels_task1_bin))

#test_labels_task1_bin = Data_Preprocessing.preprocessing_task1_binary(test_labels_task1)


'''train_labels_task2_bin = Data_Preprocessing.preprocessing_task1_binary(train_labels_task2)
test_labels_task2_bin = Data_Preprocessing.preprocessing_task1_binary(test_labels_task2)
'''

model = Model.Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.BCELoss()

model.train()

train_evaluation += model.train_x1(features=train_features_task1[:1],labels=train_labels_task1_bin[:1],loss_function= criterion,optimizer= optimizer,epochs= 1)

train_evaluation += model.train_x2(features=train_features_task2[:1],labels=train_labels_task2_bin[:1],loss_function= criterion,optimizer= optimizer,epochs= 1)

#train_evaluation += model.train_x1(features=train_features_task1,labels=train_labels_task1_bin,loss_function= criterion,optimizer= optimizer,epochs= 20)

#train_evaluation += model.train_x2(features=train_features_task2,labels=train_labels_task2_bin,loss_function= criterion,optimizer= optimizer,epochs= 20)

test_evaluation += model.test_model_x2(features=train_features_task1[:1], labels=train_labels_task1_bin[:1], loss_function=criterion)
#test_evaluation += model.test_model_x1(features=test_features_task2, labels=test_labels_task2_bin, loss_function=criterion)

print(test_evaluation, train_evaluation)