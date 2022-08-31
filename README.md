## Plagiarism Detection with a Multi-Task Perspective

The project was part of the course Information Retrieval held in the Bergische University Wuppertal as part of the Master in Industrial Engineering with Computer Science. 
In this project we developed a Multi-Task Learning model, which can learn different tasks in the area of plagiarism detection. For that we explored the field of multi-task learning for plagiarism with neural language models. The aim of the project was to develop a model that can train more than one task in the field of plagiarism detection based on previous state-of-the-art work.

# Motivation

Through the grogress made in the field of NLP plagiarism detection have expirienced a lot of success. With this project we tried to find a way to improve the learning of different tasks on the field of plagiarism detection. One way to improve the training of such neural language models is the usage of multi-task learning models.

# Features

For the tasks we used the datasets of the PAN-Paper 20 "Overview of the Cross-Domain Authorship Verification Task at PAN 2020" (https://pan.webis.de/downloads/publications/papers/kestemont_2020.pdf) and of the PAN-Paper 21 "Overview if the Style Change Detection Task at PAN 2021 (https://pan.webis.de/downloads/publications/papers/zangerle_2021.pdf) 


# Code examples

Include very short code examples that show what the project does as concisely as possible. Developers should be able to figure out how your project solves their problem by looking at the code examples. Make sure the API you are showing off is intuitive, and that your code is short and concise. See the news-please project for example.

# Installation

Provide step-by-step examples and descriptions of how to set up a development environment.
Describe and show how to run the tests with code examples.

For the project to run you need to install the BERT model, a bidirectional transformer. 
Please run: `from transformers import BertModel`
Also you need a current version of Pytorch which you can find here:
`https://pytorch.org`


# How use and extend the project? 

The next step of this project would be to debug the current network so that you can get a valid result if the multi-task learning model is working as intented. Because of time issues, we couldn´t finalise the training aspect of the project, so that we can figure if the learning of two task in different variants will get better results.
Furthermore it would be intereseting to add a third task to the model to see if the results would be even better. For that we theoretically thought about using the task of author-profiling with the dataset of PAN 18 "Overview of the 6th Author Profiling Task at PAN 20218: Multimodal Gender Identification in Twitter"

# Results





If you performed evaluations as part of your project, include your preliminary results that you also show in your final project presentation, e.g., precision, recall, F1 measure and/or figures highlighting what your project does. If applicable, briefly describe the dataset your created or used first before presenting the evaluated use cases and the results.

If you are about to complete your thesis, include the most important findings (precision/recall/F1 measure) and refer to the corresponding pages in your thesis document.

# License



