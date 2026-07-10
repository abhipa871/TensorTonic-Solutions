import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
    """
    Create an embedding layer.
    """
    # Your code here
    layer = nn.Embedding(vocab_size, d_model)
    return layer

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Convert token indices to scaled embeddings.
    
    """
    layer = embedding(tokens)*math.sqrt(d_model)
    
    return layer