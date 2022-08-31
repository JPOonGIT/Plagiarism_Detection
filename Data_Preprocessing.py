def preprocessing(data):
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(data, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    return encoded_input


def preprocessing_task1_binary(labels):
    new_labels = list()
    binary = list()
    count_ja = 0
    count_nein = 0
    for label in labels:
        if label['multi-author'] == 1:
            binary.append(1)
            new_labels.append([1,0])
            count_ja += 1
        else:
            binary.append(0)
            new_labels.append([0,1])
            count_nein += 1
    return new_labels


def preprocessing_task2_binary(same_author):
    new_labels = []
    binary = []
    count_ja = 0
    count_nein = 0
    values = list(same_author.values())
    for label in values:
        if label == 1:
            binary.append(1)
            new_labels.append([1,0])
            count_ja += 1
        else:
            binary.append(0)
            new_labels.append([1,0])
            count_nein += 1
    return new_labels