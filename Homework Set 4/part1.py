"""
Part 1: Explore Tokenization and Embeddings (25 pts)

Choose a short paragraph (5-10 sentences) from any article.

Tasks
Tokenize the text using a pretrained tokenizer (e.g., GPT-2 or BERT tokenizer).
Print:
Original sentence
Tokens
Token IDs
Convert the token IDs into embeddings using the model's embedding layer.
Print the embedding tensor shape.
Questions
Why can't we feed raw text directly into a neural network?
What information does an embedding capture that a token ID does not?
"""

import torch
from torch import nn
from transformers import BertTokenizer, BertModel
sentences = ["This is testing how embeddings work!", 
             "Embeddings are a way to represent words in a continuous vector space.", 
             "They capture semantic meaning and relationships between words.", 
             "Tokenization is the process of converting text into tokens.",]


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)

print("Original sentences:")
for sentence in sentences:
    print(sentence)
print("\nTokens:")
for token in inputs['input_ids']:
    print(token)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

print("\nEmbedding tensor shape:")
print(embeddings.shape)