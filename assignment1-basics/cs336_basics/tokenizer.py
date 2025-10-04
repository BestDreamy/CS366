from cs336_basics.train_bpe import process_single_chunk

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        Initialize the Tokenizer with a vocabulary, BPE merge rules, and optional special tokens.

        Args:
            vocab (dict[int, bytes]): A mapping from token IDs to byte-encoded tokens.
            merges (list[tuple[bytes, bytes]]): A list of merge operations as tuples of byte pairs.
            special_tokens (list[str] | None): Optional list of user-defined special tokens to include.
        """
        self.vocab: dict[int, bytes] = vocab
        self.vocab_img: dict[bytes, int] = {v: k for k, v in self.vocab.items()}
        self.merges: list[tuple[bytes, bytes]] = merges
        self.special_tokens: list[str] = special_tokens if special_tokens is not None else []
        self.replace_token = bytes("\uFFFD", encoding='utf-8')  # Unicode replacement character

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        Construct a Tokenizer from serialized vocabulary and merges files.

        Args:
            vocab_filepath (str): Path to the vocabulary file (from BPE training).
            merges_filepath (str): Path to the merges file (from BPE training).
            special_tokens (list[str] | None): Optional list of special tokens to include.

        Returns:
            Tokenizer: A Tokenizer instance initialized with the given files.
        """
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = token_str.encode("utf-8")  # store as bytes

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode an input string into a list of token IDs using the BPE algorithm.

        Args:
            text (str): The input text to tokenize.

        Returns:
            list[int]: A list of token IDs representing the encoded text.
        """
        token_ids: list[int] = []
        words: list[bytes]
        
        words, _ = process_single_chunk(text, self.special_tokens, False, False)
        for word in words:
            # print(word)
            if word.decode('utf-8') in self.special_tokens:
                token_ids.append(self.vocab_img[word])
                continue

            word = [bytes([b]) for b in word]  # Convert each byte to bytes of length 1
            for pair in self.merges:
                new_word: list[bytes] = []
                a, b = pair
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                        token = a + b
                        i += 2
                    else:
                        token = word[i]
                        i += 1

                    new_word.append(token)
                word = new_word
            
            # print(word)
            for token in word:
                token_ids.append(self.vocab_img[token])
        
        return token_ids


    def encode_iterable(self, iterable: list[str]) -> iter:
        """
        Lazily encode an iterable of strings into a stream of token IDs.

        Useful for memory-efficient tokenization of large datasets.

        Args:
            iterable (list[str]): An iterable of strings (e.g., lines from a file).

        Returns:
            iter: A generator that yields token IDs one at a time.
        """
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = b''

        for token_id in token_ids:
            token = self.vocab.get(token_id, self.replace_token)  # Use replacement char if ID not found

            tokens += token

        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 


if __name__ == "__main__":
    vocab = {
        0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at',
        11: b'T', 12: b's', 13: b'.', 14: b'<|endoftext|>'
    }
    merges = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens=special_tokens)
    token_ids:list[int] = tokenizer.encode("<|endoftext|>The cat sat <|endoftext|>at the cata.<|endoftext|>")
    print(token_ids)
    token: str = tokenizer.decode(token_ids)
    print(token)
