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
