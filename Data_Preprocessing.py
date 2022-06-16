
def preprocessing(data):
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_input = tokenizer(data, return_tensors='pt', padding=True, truncation=True)
    
    return encoded_input
