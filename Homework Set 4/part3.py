"""
Part 3: Visualize Attention (30 pts)

Choose a sentence such as:

"The cat sat on the mat because it was tired."

Use your attention implementation (or a pretrained model if covered in class) to produce an attention matrix.

Visualize it with matplotlib.imshow().

Questions
Which words receive the most attention?
Which words attend to themselves?
Does the pattern match your intuition?

"""
import torch
from torch import nn
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

sentence = "The cat sat on the mat because it was tired."

inputs = tokenizer(sentence, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

# Get the attention matrix for the first layer and first head
attention_matrix = attentions[0][0][0].detach().numpy()

# Visualize the attention matrix
plt.imshow(attention_matrix, cmap='hot', interpolation='nearest')
plt.title('Attention Matrix')
plt.xlabel('Keys')
plt.ylabel('Queries')
plt.show()

# Visualize which words receive the most attention
attention_sum = attention_matrix.sum(axis=0)
plt.bar(range(len(attention_sum)), attention_sum)
plt.title('Attention Received by Each Word')
plt.xlabel('Word Index')
plt.ylabel('Attention Score')
plt.xticks(range(len(attention_sum)), tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), rotation=45)
plt.show()