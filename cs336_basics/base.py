import os

from collections import defaultdict

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges

    def pretokenize(self, input_path: str | os.PathLike) -> list[str]:
        raise NotImplementedError

    def train(
            self, 
            train_data: list[str], 
            vocab_size: int, 
            special_tokens: list[str]
            ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        raise NotImplementedError

    def encode(self, input_text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError

def merge(train_bytes: list[list[int]], pair: tuple[int, int], newtoken: int) -> list[list[int]]:
    """Merge all occurrences of `pair` (two token ids) into a single newtoken id.

    train_bytes is a list of token-id sequences. This returns a new list of
    sequences where every adjacent occurrence of `pair` is replaced by
    `newtoken` (an int).
    """
    new_train_bytes: list[list[int]] = []
    for token_bytes in train_bytes:
        i = 0
        merged_tokens: list[int] = []
        while i < len(token_bytes):
            if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i + 1]) == pair:
                merged_tokens.append(newtoken)
                i += 2
            else:
                merged_tokens.append(token_bytes[i])
                i += 1
        new_train_bytes.append(merged_tokens)
    return new_train_bytes