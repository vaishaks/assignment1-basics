import os

from collections import defaultdict

class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges

    def pretokenize(self, input_path: str | os.PathLike) -> list[str]:
        raise NotImplementedError

    def train(self, train_data: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        raise NotImplementedError

    def encode(self, input_text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError
    
def count_pairs(train_bytes: list[list[bytes]]) -> dict[tuple[int, int], int]:
    byte_pairs: list[tuple[int, int]] = []
    for token_bytes in train_bytes:
        for p1, p2 in zip(token_bytes[:-1], token_bytes[1:]):
            byte_pairs.append((p1, p2))

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for p in byte_pairs:
        pair_counts[p] += 1
    return pair_counts


def merge(train_bytes: list[list[bytes]], pair: tuple[int, int], newtoken: int) -> list[list[bytes]]:
    new_train_bytes: list[list[bytes]] = []
    for token_bytes in train_bytes:
        i = 0
        merged_bytes: list[bytes] = []
        while i < len(token_bytes):
            if i < len(token_bytes) - 1 \
                and (token_bytes[i], token_bytes[i + 1]) == pair:
                merged_bytes.append(newtoken)
                i += 2
            else:
                merged_bytes.append(token_bytes[i])
                i += 1
        new_train_bytes.append(merged_bytes)
    return new_train_bytes