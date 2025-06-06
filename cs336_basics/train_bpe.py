import os
from .pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import regex
from collections import defaultdict

def process_single_chunk(
    args: tuple[int, int, str|bytes, list[str]]
) -> list[list[bytes]]:
    """
    Convert a chunk of text into bytes.
    """
    start, end, input_path, special_token = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")

    # Regex just support str
    special_token_str = special_token[0]
    special_escape_token = f'({regex.escape(special_token_str)})'

    segments = regex.split(special_escape_token, chunk_text) # regex.split(str, str)

    byte_lst: list[list[bytes]] = []

    for i in range(len(segments)):
        if segments[i] == special_token_str:
            byte_lst.append([special_token_str.encode("utf-8")])
        else:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for match in regex.finditer(PAT, segments[i]):
                match_str = match.group(0)
                byte_lst.append([bytes([c]) for c in match_str.encode("utf-8")])

    return byte_lst


def train_bpe_model(
    input_path: str,
    vocab_size: int,
    special_token: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_token_bytes = special_token[0].encode("utf-8")

    with open(input_path, "rb") as f:
        num_processes = mp.cpu_count()
        # print(f"{num_processes} processes available for chunking.")
        chunk_boundaries = find_chunk_boundaries(
            f, num_processes, special_token_bytes
        )
    # print(chunk_boundaries)
    # with open(input_path, "rb") as f:
    #     for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
    #         f.seek(start)
    #         chunk = f.read(end - start)
    #         print(f"Chunk [{start}:{end}]:", chunk)

    tokens_lst: list[list[bytes]] = []

    args = [(start, end, input_path, special_token) for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]

    with mp.Pool(num_processes) as pool:
        for segments in pool.imap(process_single_chunk, args):
            # print(segments)
            tokens_lst.extend(segments)
    # print(tokens_lst)

    vocab_dict = {
        i: special_token_bytes if i == 0 else bytes([i - 1]) for i in range(257)
    }
    merge_lst = []

    vocab_size = min(vocab_size, 1100)

    for idx in range(len(vocab_dict), vocab_size):
        count = defaultdict(int)
        # Find the most frequent byte pair in the tokens_lst
        for it in tokens_lst:
            for i in range(len(it) - 1):
                pair = (it[i], it[i + 1])
                count[pair] += 1
        if not count:
            break
        # max_pair = max(count, key=count.get)
        max_pair = max(count.items(), key=lambda x: (x[1], x[0]))[0]

        new_vocab = max_pair[0] + max_pair[1]
        vocab_dict[idx] = new_vocab
        merge_lst.append(max_pair)

        new_tokens_lst = []
        for it in tokens_lst:
            i = 0
            new_token = []
            while i < len(it):
                if i < len(it) - 1 and (it[i], it[i + 1]) == max_pair:
                    new_token.append(new_vocab)
                    i += 2
                else:
                    new_token.append(it[i])
                    i += 1
            new_tokens_lst.append(new_token)

        tokens_lst = new_tokens_lst
    
    return vocab_dict, merge_lst


# if __name__ == '__main__':
#     split_token = ["<|endoftext|>"]
#     data = split_token[0].encode("utf-8").join([
#         b"Hello world!",
#         b"Python is great.",
#         b"Test chunking.",
#         b"End of file."
#     ])
#     with open("test_data.bin", "wb") as f:
#         f.write(data)
    
#     train_bpe_model(
#         input_path="test_data.bin",
#         vocab_size=260,
#         special_token=split_token,
#     )
    # os.remove("test_data.bin")