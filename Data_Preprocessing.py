
def preprocessing(data):
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(data, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

    return encoded_input
