import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        vocab = set()
        for text in texts:
            words = text.lower().split()
            vocab.update(words)     
        vocab = sorted(list(vocab))
        special_order = [self.pad_token , self.unk_token , self.bos_token, self.eos_token]
        vocab = special_order + vocab 
        self.vocab_size = len(vocab)
        self.word_to_id = {word:id for id, word in enumerate(vocab)}
        self.id_to_word = {id:word for id, word in enumerate(vocab)}    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.lower().split()
        arr = [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words]
        return arr
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        return " ".join(
            self.id_to_word.get(token_id, self.unk_token)
            for token_id in ids
        )
