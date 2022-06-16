from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open("Data.txt", "r") as f:
    text = f.readlines()

model = BertModel.from_pretrained("bert-base-uncased")

'''
output = model(**encoded_input)

print(output)
      '''
