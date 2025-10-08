import regex as re
from collections import defaultdict, Counter
import multiprocessing as mp
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_tokens: list[bytes]
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert all(isinstance(token, bytes) for token in split_special_tokens), (
        "All special tokens must be represented as bytestrings"
    )

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

            # Find the first occurrence of any special token in the mini chunk
            found_at = -1
            for token in split_special_tokens:
                token_index = mini_chunk.find(token)
                if token_index != -1 and (found_at == -1 or token_index < found_at):
                    found_at = token_index

            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def split_by_special(
    text: str, 
    special_tokens: list[str], 
    drop_special=True
) -> list[str]:
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True)
    pattern: str = "|".join(re.escape(tok) for tok in special_tokens)
    if not drop_special:
        pattern = f"({pattern})"

    text_chunks: list[str] = re.split(pattern, text)
    return [text_chunk for text_chunk in text_chunks if text_chunk]

def process_single_chunk(
    text: str,
    special_tokens: list[str],
    drop_special=True,
    word_is_unique=True
) -> tuple[list[bytes], Counter[bytes]]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # Pre-Tokenization
    # 1. Split by special tokens
    text_chunks: list[str] = split_by_special(text, special_tokens, drop_special)

    # 2. Statistics words on each chunk
    words: list[bytes] = []
    word2count: Counter[bytes] = Counter()

    for text_chunk in text_chunks:
        if text_chunk in special_tokens:
            word = text_chunk.encode("utf-8")
            word2count[word] += 1
            if word2count[word] == 1 or not word_is_unique:
                words.append(word)
            continue

        partial_words = re.findall(PAT, text_chunk)

        for word in partial_words:
            word = word.encode("utf-8")
            word2count[word] += 1

            if word2count[word] == 1 or not word_is_unique:
                words.append(word)
    
    return words, word2count
    
def process_single_chunk_from_file(
    args: tuple[int, int, str, list[str]]
) -> tuple[list[bytes], Counter[bytes]]:
    start, end, input_path, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes: bytes = f.read(end - start)

    chunk_text: str = chunk_bytes.decode("utf-8", errors="ignore")

    return process_single_chunk(chunk_text, special_tokens)


def BPETokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    special_byte_tokens: list[bytes] = [it.encode("utf-8") for it in special_tokens]

    # Step 1: Split the file according to special tokens
    num_process: int = mp.cpu_count()
    with open(input_path, "rb") as f:
        chunk_boundaries: list[int] = find_chunk_boundaries(
            f, num_process, special_byte_tokens
        )

    # Step 2: Count words in each chunk(Pre-Tokenization)
    args = [(start, end, input_path, special_tokens) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]

    words: list[bytes] = []
    word2count: Counter[bytes] = Counter()

    with mp.Pool(num_process) as pool:
        for single_chunk_words, single_chunk_word2count in pool.imap(process_single_chunk_from_file, args):
            for word in single_chunk_words:
                if word2count[word] == 0:
                    words.append(word)
                
                word2count[word] += single_chunk_word2count[word]

    # Step 3: Build BPE vocabulary according to the word frequency

    # Initial vocabulary with all special tokens and single byte tokens
    num_special_token = len(special_tokens)
    vocab_dict: dict[int, bytes] = {
        i: special_byte_tokens[i] if i < num_special_token
                                else bytes([i - num_special_token]) for i in range(256 + num_special_token)
    }
    merge_lst = []

    # Merge tokens according to the BPE algorithm
    word2tokens: dict[bytes, list[bytes]] = {
        word: [bytes([b]) for b in word] for word in words
    }

    for idx in range(len(vocab_dict), vocab_size):
        # Each iteration, find the most frequent byte pair
        count = defaultdict(int)

        # Find the most frequent byte pair in the tokens_lst
        for word in words:
            tokens: list[bytes] = word2tokens[word]
            for i in range(len(tokens) - 1):
                pair: tuple[bytes, bytes] = (tokens[i], tokens[i + 1])
                if not (isinstance(pair, tuple) and len(pair) == 2  and word2count[word] > 0):
                    raise TypeError("Invalid pair or word count")
                count[pair] += word2count[word]

        # Too many vocab, but too few words
        if not count:
            break

        # Find the most frequent pair (with lexicographical tie-break)
        ans_pair: tuple[bytes, bytes] = max(count.items(), key=lambda x: (x[1], x[0]))[0]

        new_vocab: bytes = ans_pair[0] + ans_pair[1] # tuple to bytes
        vocab_dict[idx] = new_vocab
        merge_lst.append(ans_pair)

        new_word2tokens: dict[bytes, list[bytes]] = {}
        # Merge the most frequent pair in the word2tokens dict
        for word, tokens in word2tokens.items():
            i = 0
            new_tokens: list[bytes] = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == ans_pair:
                    new_tokens.append(new_vocab)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word2tokens[word] = new_tokens

        word2tokens = new_word2tokens
    
    return vocab_dict, merge_lst


if __name__ == '__main__':
    
    vocab_dict, merge_lst = BPETokenizer(
        input_path="/root/work/CS366/tests/fixtures/tinystories_oneline.txt",
        vocab_size=260,
        special_tokens=["<|endoftext|>", "<|pad|>"],
    )

    print(vocab_dict)
    print(merge_lst)