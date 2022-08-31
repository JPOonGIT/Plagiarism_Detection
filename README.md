## Plagiarism Detection with a Multi-Task Perspective

The project was part of the course Information Retrieval held in the Bergische University Wuppertal as part of the Master in Industrial Engineering with Computer Science. 
In this project we developed a Multi-Task Learning model, which can learn different tasks in the area of plagiarism detection. For that we explored the field of multi-task learning for plagiarism with neural language models. The aim of the project was to develop a model that can train more than one task in the field of plagiarism detection based on previous state-of-the-art work.

# Motivation

Through the grogress made in the field of NLP plagiarism detection have expirienced a lot of success. With this project we tried to find a way to improve the learning of different tasks on the field of plagiarism detection. One way to improve the training of such neural language models is the usage of multi-task learning models.

# Features

For the tasks we used the datasets of the PAN-Paper 20 "Overview of the Cross-Domain Authorship Verification Task at PAN 2020" (https://pan.webis.de/downloads/publications/papers/kestemont_2020.pdf) and of the PAN-Paper 21 "Overview if the Style Change Detection Task at PAN 2021 (https://pan.webis.de/downloads/publications/papers/zangerle_2021.pdf) 


# Code examples

We used a couple of different hard-coded functions for thecimport of data in the model:
- For the Input of the data we used some functions which are explained below ([Doc_import.py](https://github.com/JPOonGIT/Plagiarism_Detection/blob/2850e77301d446fc47a4a51945addd032afcd873/Doc_import.py)):
- The function [doc_import()](https://github.com/JPOonGIT/Plagiarism_Detection/blob/2850e77301d446fc47a4a51945addd032afcd873/Doc_import.py#L1) is a testfunction, which need a filename as input and will return the text of the file, without any paragraphs.
- The function [import_pan21()](https://github.com/JPOonGIT/Plagiarism_Detection/blob/27c7b38f5da789420e1fb8f258329d002dfa97cd/Doc_import.py#L11) is the import function of the first dataset. It needs a base path as input where the training and test data is located. The Output will be four lists with the training/test labels and features
- The function [import_verification()](https://github.com/JPOonGIT/Plagiarism_Detection/blob/27c7b38f5da789420e1fb8f258329d002dfa97cd/Doc_import.py#L56)  is the import function for the second dataset. It needs also the base path of the training and test data as input. The Output will be four lists with the training/test labels and features
- The [text_preprocessing()](https://github.com/JPOonGIT/Plagiarism_Detection/blob/27c7b38f5da789420e1fb8f258329d002dfa97cd/Doc_import.py#L102) is function for the second task. It needs the paths of the trainings or test data as input. The output will be a dictionary with the `ids` of the texts paired with the label `same` which indicates if a texts is from the same author. So the output will be ID:"0" or ID:"1". Additionally you the function returns list with the texts of the json file, if the id is in same_author.

We also use different functions to preprocess the data for the model:
- The function [preprocessing()](https://github.com/JPOonGIT/Plagiarism_Detection/blob/4f4ff1b7887a15bfd4ae559c8cf66a9576c72240/Data_Preprocessing.py#L1) gets as input an text and returns a three tensors for the bert model input. The function is based on the bert tokenizer.
- The function [processing_taks1_binary()](https://github.com/JPOonGIT/Plagiarism_Detection/blob/4f4ff1b7887a15bfd4ae559c8cf66a9576c72240/Data_Preprocessing.py#L10) and [processing_task2_binary](https://github.com/JPOonGIT/Plagiarism_Detection/blob/fce24231be0bf3f680de1a662c06c81de2e46998/Data_Preprocessing.py#L27) both transform the labels from the supervised data into binary format and returns it.


The Model.py file includes the functions relating to the model, like training testing and so on
- Function [__init__(self)](https://github.com/JPOonGIT/Plagiarism_Detection/blob/978045b40af11fc632b6a2351224176dff611331/Model.py#L9) initialize the model with all the needed layer if the class gets called
- Function [forward_x1 and forward_x2](https://github.com/JPOonGIT/Plagiarism_Detection/blob/978045b40af11fc632b6a2351224176dff611331/Model.py#L25) describes the path through the model for a training step. Depending on which task you want to train the right function gets called in the train function.
- Function [train_x1 train_x2](https://github.com/JPOonGIT/Plagiarism_Detection/blob/978045b40af11fc632b6a2351224176dff611331/Model.py#L45) are the two functions which do the actual training step. For that you give the function a list with the labels and the training features. The function iterates over the lenght of both. Also it is importont to define a loss function and a optimizer for it. The last parameter to choose is the epoch parameter. With that you can choose how often a task gets with the same data trained. The function returns a list with the loss of all training samples. 
- Function [test_model_x1 test_model_x2](https://github.com/JPOonGIT/Plagiarism_Detection/blob/978045b40af11fc632b6a2351224176dff611331/Model.py#L87) with this function it is possible to calculate a loss with the test data and compare it with the training evaluation. The function returns a list with the loss of all test samples.

# Installation

For the project to run you need to install the BERT model, a bidirectional transformer. 
Please run: `from transformers import BertModel`
Also you need a current version of Pytorch which you can find here:
`https://pytorch.org`


# How use and extend the project? 

The next step of this project would be to debug the current network so that you can get a valid result if the multi-task learning model is working as intented. Because of time issues, we couldn´t finalise the training aspect of the project, so that we can figure if the learning of two task in different variants will get better results.
Furthermore it would be intereseting to add a third task to the model to see if the results would be even better. For that we theoretically thought about using the task of author-profiling with the dataset of PAN 18 "Overview of the 6th Author Profiling Task at PAN 20218: Multimodal Gender Identification in Twitter"

# Results

Although we had some results after the final testing, these results are not representative or evaluable. But these result show that the model is working, in the following screenshot you can see the output we got for the final result.
![Bildschirmfoto 2022-08-31 um 21 54 28](https://user-images.githubusercontent.com/86957713/187770121-6347707b-61d3-4ca5-932d-2b7ccfe1e614.png)
