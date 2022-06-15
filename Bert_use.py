from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

text = "Jan and Alex are too good for information retrieval. Lars is also a pretty good guy."
encoded_input = tokenizer(text, return_tensors='pt')

#output = model(**encoded_input)

#output_decoded = tokenizer.decode(output)

#print(output_decoded)

print(encoded_input)