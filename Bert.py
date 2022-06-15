from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

with open("Data.txt", "r") as f:
    text = f.readlines()

encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

output = model(**encoded_input)

print(output)
