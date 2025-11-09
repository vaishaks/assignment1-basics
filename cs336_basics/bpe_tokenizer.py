import os
import regex as re 

from multiprocessing import Pool
from typing import BinaryIO

from .base import Tokenizer

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT_TOKEN = "<|endoftext|>"

class ListNode(object):
    def __init__(self, x: int):
        self.val = x
        self.next = None
        self.prev = None

    @classmethod
    def from_list(cls, list_of_ints: list[int]) -> 'ListNode':
        if list_of_ints:
            n = cls(list_of_ints[0])
            next = ListNode.from_list(list_of_ints[1:])
            n.next = next
            if next:
                next.prev = n
            return n
        
    def print_list(self) -> None:
        n = self
        buffer = []
        while n:
            buffer.append(str(n.val))
            n = n.next
        s = "->".join(buffer)
        print(s)

    def to_list(self) -> list[int]:
        n = self
        buffer = []
        while n:
            buffer.append(n.val)
            n = n.next
        return buffer

class BPETokenizer(Tokenizer):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        super().__init__(vocab, merges)

    @staticmethod
    def _process_chunk(args) -> list[str]:
        start, end, input_path = args        
        chunk_tokens = []
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Split on EOT_TOKEN but preserve it as separate pretokens
            parts = chunk.split(EOT_TOKEN)
            for i, part in enumerate(parts):
                # Apply regex pretokenization to the text part
                for pretoken in re.finditer(PAT, part):
                    chunk_tokens.append(pretoken.group(0))
                # Insert EOT_TOKEN as a pretoken between parts (but not after the last part)
                if i < len(parts) - 1:
                    chunk_tokens.append(EOT_TOKEN)
        return chunk_tokens

    def pretokenize(self, input_path: str | os.PathLike) -> list[str]:
        num_processes = os.cpu_count() or 4  # Use CPU count or fallback to 4
        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, num_processes, EOT_TOKEN.encode("utf-8"))
        
        # Create list of (start, end, input_path) tuples for each chunk
        chunk_args = [(start, end, input_path) \
                      for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            chunk_results = pool.map(self._process_chunk, chunk_args)
        
        # Flatten results from all processes
        pretokenized_train_data = []
        for chunk_tokens in chunk_results:
            pretokenized_train_data.extend(chunk_tokens)
            
        return pretokenized_train_data

    def _init_vocab(self, special_tokens: list[str]) -> dict[int, bytes]:
        vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")
        return vocab

    def _build_sequences(self, train_data: list[str], vocab: dict[int, bytes]) -> list[ListNode]:
        # Build a mapping from special token bytes to their vocab IDs
        special_token_to_id = {}
        for token_id, token_bytes in vocab.items():
            if token_id >= 256:  # Special tokens are added after byte tokens
                special_token_to_id[token_bytes] = token_id
        
        # Preprocess the training data into a list of token ids (ints)
        corpus: list[list[int]] = []
        for pretoken in train_data:
            pretoken_bytes = pretoken.encode("utf-8")
            # If this pretoken is a special token, use its vocab ID directly (atomic unit)
            if pretoken_bytes in special_token_to_id:
                corpus.append([special_token_to_id[pretoken_bytes]])
            else:
                # Regular pretoken: break into bytes
                corpus.append(list(pretoken_bytes))

        # Create linked list sequences for each pretoken
        sequences: list[ListNode] = []
        for seq in corpus:
            sequence = ListNode.from_list(seq)
            sequences.append(sequence)

        return sequences        
    
    def _index_pairs(self, sequences: list[ListNode]) -> dict[tuple[int, int], set[ListNode]]:
        pair_positions: dict[tuple[int, int], set[ListNode]] = {}
        for seq in sequences:
                token = seq
                while token and token.next:
                    pair = (token.val, token.next.val)
                    if pair not in pair_positions:
                        pair_positions[pair] = set()
                    pair_positions[pair].add(token)
                    token = token.next  
        return pair_positions
    
    def _apply_merge(
        self,
        pair: tuple[int, int],
        tokenid: int,
        pair_positions: dict[tuple[int, int], set[ListNode]]
        ) -> None:
            ocurrences = list(pair_positions[pair])
            for pos in ocurrences:
                # We will merge pos and pos.next into a single node with value tokenid
                # All it's neighbors are going to be affected by this merge and we need to 
                # update pair_positions by first removing old references.
                for neighbor in (pos.prev, pos.next):
                    if neighbor and neighbor.next:
                        old_pair = (neighbor.val, neighbor.next.val)
                        if old_pair in pair_positions:
                            pair_positions[old_pair].discard(neighbor)
                # Merge nodes
                old_node = pos.next
                pos.val = tokenid
                pos.next = old_node.next
                if old_node.next:
                    old_node.next.prev = pos

                # Add new pairs formed by the merge
                if pos.prev:
                    new_pair = (pos.prev.val, pos.val)
                    if new_pair not in pair_positions:
                        pair_positions[new_pair] = set()
                    pair_positions[new_pair].add(pos.prev)
                if pos.next:
                    new_pair = (pos.val, pos.next.val)
                    if new_pair not in pair_positions:
                        pair_positions[new_pair] = set()
                    pair_positions[new_pair].add(pos)
            # Remove merged pair from pair_positions
            if pair in pair_positions:
                del pair_positions[pair]

    def train(
            self,
            train_data: list[str],
            vocab_size: int,
            special_tokens: list[str]
            ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Initialize base vocab (single-byte tokens)
        vocab: dict[int, bytes] = self._init_vocab(special_tokens)
        merges: list[tuple[bytes, bytes]] = []
        sequences: list[ListNode] = self._build_sequences(train_data, vocab)
        
        # Get set of special token IDs to skip during merging
        special_token_ids = {tid for tid, token_bytes in vocab.items() if tid >= 256}

        # Pre-compute the positions of all adjacent pairs in the sequences
        pair_positions: dict[tuple[int, int], set[ListNode]] = self._index_pairs(sequences)

        # Determine how many merges to perform given the desired final vocab size
        num_merges = max(0, vocab_size - len(vocab))
        # Perform BPE merges.
        for idx in range(num_merges):
            # Build frequency map, excluding pairs that involve special tokens
            pair_frequencies: dict[tuple[int, int], int] = {
                pair: len(pos) for pair, pos in pair_positions.items() 
                if len(pos) >= 2 and pair[0] not in special_token_ids and pair[1] not in special_token_ids
            }
            if not pair_frequencies:
                break
            # Lexicographical ordering of the most frequent pair
            def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
                # Sort primarily by count, then by lexicographical order of bytes
                return (pair_frequencies[pair], (vocab[pair[0]], vocab[pair[1]]))
            # Find the most frequent (and lexicographically largest) pair
            max_pair = max(pair_frequencies, key=rank)
            freq = pair_frequencies[max_pair]
            if freq < 2:
                break

            # Mint new token
            tokenid = len(vocab)
            # Update vocab with the concatenation of the two merged bytes as a new token
            vocab[tokenid] = vocab[max_pair[0]] + vocab[max_pair[1]]
            # Save the merge
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

            # Apply the merge and update pair positions
            self._apply_merge(max_pair, tokenid, pair_positions)

        self.vocab = vocab
        self.merges = merges
        return vocab, merges
    
    def encode(self, input_text: str) -> list[int]:
        # Pretokenize input into strings according to PAT
        pretok_text = re.findall(PAT, input_text)
        # Convert each pretoken string into a list of UTF-8 byte values (ints)
        input_text_bytes = [list(pretoken.encode("utf-8")) for pretoken in pretok_text]
        # Build a reverse mapping from byte sequence (bytes) to token id (int)
        byte_to_id = {b: tid for tid, b in self.vocab.items()}
        merged = []
        # Process each pretoken's byte-token sequence independently
        for tokens in input_text_bytes:
            # tokens is a list[int] representing initial single-byte token ids (0-255)
            for idx, merge_pair in enumerate(self.merges):
                # Each merge_pair is stored as (bytes_left, bytes_right).
                left_bytes, right_bytes = merge_pair
                # Use the reverse map to find the token ids.
                left_id = byte_to_id.get(left_bytes)
                right_id = byte_to_id.get(right_bytes)
                new_token_id = 256 + idx
                i = 0
                merged_tokens: list[int] = []
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == (left_id, right_id):
                        merged_tokens.append(new_token_id)
                        i += 2
                    else:
                        merged_tokens.append(tokens[i])
                        i += 1              
                if merged_tokens == tokens:
                    break  # No more merges can be applied
                tokens = merged_tokens
            merged.extend(tokens)
        return merged

    def decode(self, token_ids: list[int]) -> str:
        byte_array: list[bytes] = []
        for token in token_ids:
            byte_array.append(self.vocab[token])
        return b"".join(byte_array).decode("utf-8")
    
    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:        
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
