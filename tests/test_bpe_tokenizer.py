import pathlib

from .solution import bpe_tokenizer

FIXTURES_PATH = (pathlib.Path(__file__).resolve().parent) / "fixtures"

def test_pretokenize():
    tokenizer = bpe_tokenizer.BPETokenizer(vocab={}, merges=[])
    input_path = FIXTURES_PATH / "tinystories_sample.txt"
    pretokenized_data = tokenizer.pretokenize(input_path)
    print(pretokenized_data)
    assert len(pretokenized_data) > 0
    assert pretokenized_data[:10] == ['\n', 'Once', ' upon', ' a', ' time', ' there', ' was', ' a', ' little', ' boy']
