"""
Part 2: Implement Scaled Dot-Product Attention (45 pts)

Implement the attention equation yourself using PyTorch tensor operations.

Given matrices:

Q (queries)
K (keys)
V (values)

Compute:

Attention scores:
QK
T
Scale by
d
k
	​

	​

Apply Softmax.
Multiply by V.

Do not use torch.nn.MultiheadAttention.

Print:

Attention scores
Attention weights
Final output
Reflection

Answer:

Why is Softmax applied?
What do the attention weights represent?

"""
import torch
from torch import nn

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    # Apply Softmax to get attention weights
    attention_weights = torch.nn.functional.softmax(scores, dim=-1)
    # Multiply by V to get the final output
    output = torch.matmul(attention_weights, V)
    return scores, attention_weights, output