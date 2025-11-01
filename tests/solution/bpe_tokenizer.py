import os
import regex as re 
from typing import BinaryIO
from multiprocessing import Pool
from .base import Tokenizer

class BPETokenizer(Tokenizer):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]]):
        super().__init__(vocab, merges)

    @staticmethod
    def _process_chunk(args) -> list[str]:
        start, end, input_path = args
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        chunk_tokens = []
        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = "|".join(chunk.split("<|endoftext|>"))
            for pretoken in re.finditer(PAT, chunk):
                chunk_tokens.append(pretoken.group(0))
        return chunk_tokens

    def pretokenize(self, input_path: str | os.PathLike) -> list[str]:
        num_processes = os.cpu_count() or 4  # Use CPU count or fallback to 4
        with open(input_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # Create list of (start, end, input_path) tuples for each chunk
        chunk_args = [(start, end, input_path) for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            chunk_results = pool.map(self._process_chunk, chunk_args)
        
        # Flatten results from all processes
        pretokenized_train_data = []
        for chunk_tokens in chunk_results:
            pretokenized_train_data.extend(chunk_tokens)
            
        return pretokenized_train_data

    def train(self, train_data: list[str]):
        raise NotImplementedError
    
    def encode(self, input_text: str) -> list[int]:
        raise NotImplementedError
    
    def decode(self, token_ids: list[int]) -> str:
        raise NotImplementedError
    
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
