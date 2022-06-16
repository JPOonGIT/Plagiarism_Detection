def doc_import():

    docs = input("Input filename: ")
    while len(docs) < 1: break
    with open(docs, 'r') as f:
        for line in f:
            line = f.read().replace('\n', '')
    return line