import re
from collections import Counter
import torch

class Tokenizer:
    def __init__(self, vocab=None, unk_token="<unk>", pad_token="<pad>", max_vocab_size=10000):
        self.vocab = vocab or {}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.max_vocab_size = max_vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.build_vocab()

    def build_vocab(self):
        if not self.vocab:
            self.token_to_id = {
                self.unk_token: 0,
                self.pad_token: 1
            }
            self.id_to_token = {0: self.unk_token, 1: self.pad_token}
        else:
            # Add special tokens to the vocab
            self.vocab = {self.unk_token: 0, self.pad_token: 1, **self.vocab}
            self.token_to_id = self.vocab
            self.id_to_token = {id: token for token, id in self.vocab.items()}

    def tokenize(self, text):
        """
        Tokenizes a string of text into tokens.
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        """
        Converts a list of tokens to their corresponding IDs.
        """
        return [self.token_to_id.get(token, self.token_to_id[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids):
        """
        Converts a list of token IDs to their corresponding tokens.
        """
        return [self.id_to_token.get(id, self.unk_token) for id in ids]

    def text_to_tensor(self, text, max_length=None):
        """
        Converts a string of text to a tensor of token IDs.
        """
        tokens = self.tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)
        
        if max_length:
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            else:
                token_ids += [self.token_to_id[self.pad_token]] * (max_length - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long)

    def tensor_to_text(self, tensor):
        """
        Converts a tensor of token IDs back to a string of text.
        """
        token_ids = tensor.tolist()
        tokens = self.convert_ids_to_tokens(token_ids)
        return ' '.join(tokens)

    def update_vocab(self, texts):
        """
        Updates the vocabulary based on a list of texts.
        """
        counter = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)
        
        most_common_tokens = counter.most_common(self.max_vocab_size - 2)  # Reserve space for <unk> and <pad>
        new_vocab = {token: idx + 2 for idx, (token, _) in enumerate(most_common_tokens)}
        
        self.vocab.update(new_vocab)
        self.build_vocab()

