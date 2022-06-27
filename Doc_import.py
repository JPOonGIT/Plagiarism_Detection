def doc_import():

    docs = input("Input filename: ")
    while len(docs) < 1: break
    with open(docs, 'r') as f:
        for line in f:
            line = f.read().replace('\n', '')
    return line


def import_pan21(base_path):
    import os
    import json

    test_train = os.listdir(base_path)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for dic in test_train:
        path = base_path + "/" + dic
        data_path = os.listdir(path)

        if dic == test_train[0]:
            for i, file_name in enumerate(data_path[0:int(len(data_path) / 2)]):
                file_path = path + "/" + file_name
                with open(file_path, 'r', encoding='utf8') as f:
                    text = f.readlines()
                    train_features.append(text)

            for i, file_name in enumerate(data_path[int(len(data_path) / 2):int(len(data_path))]):
                file_path = path + "/" + file_name
                f = open(file_path)
                text = json.load(f)
                train_labels.append(text)

        if dic == test_train[1]:
            for i, file_name in enumerate(data_path[0:int(len(data_path) / 2)]):
                file_path = path + "/" + file_name
                with open(file_path, 'r', encoding='utf8') as f:
                    text = f.readlines()

                    test_features.append(text)

            for i, file_name in enumerate(data_path[int(len(data_path) / 2):int(len(data_path))]):
                file_path = path + "/" + file_name
                f = open(file_path)
                text = json.load(f)
                train_labels.append(text)

        return train_features, train_labels, test_features, test_labels

'''
def import_verification(base_path):
    import os
    import json

    test_train = os.listdir(base_path)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for dic in test_train:
        path = base_path + "/" + dic
        data_path = os.listdir(path)

        if dic == test_train[0]:
            for i, file_name in enumerate(data_path[0:int(len(data_path) / 2)]):
                file_path = path + "/" + file_name
                with open(file_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                    if data == "pairs":
                        text =
                        train_features.append(text)

                for i, file_name in enumerate(data_path[int(len(data_path) / 2):int(len(data_path))]):
                    file_path = path + "/" + file_name
                    f = open(file_path)
                    data = json.load(f)
                        if data ==
                            train_labels.append(text)

        if dic == test_train[1]:
            for i, file_name in enumerate(data_path[0:int(len(data_path) / 2)]):
                file_path = path + "/" + file_name
                with open(file_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                    if data == "fandoms":
                        test_features.append(text)

            for i, file_name in enumerate(data_path[int(len(data_path) / 2):int(len(data_path))]):
                file_path = path + "/" + file_name
                f = open(file_path)
                text = json.load(f)
                train_labels.append(text)
'''
