"""
bpe_tokenizer_from_scratch.py
=============================


This file demonstrates how Byte-Pair Encoding (BPE) tokenization works
from scratch.


We build two tokenizers:


1. BasicTokenizer
   - Converts the entire text directly into UTF-8 bytes.
   - Runs BPE over the complete byte sequence.
   - Merges are allowed across word / punctuation / whitespace boundaries.


2. RegexTokenizer
   - First splits text using a GPT-4-style regular expression.
   - Converts each chunk into UTF-8 bytes.
   - Runs BPE independently inside each chunk.
   - Prevents BPE merges from crossing pre-tokenization boundaries.


The workflow is:


    Text
      |
      v
    UTF-8 bytes
      |
      v
    Count adjacent pairs
      |
      v
    Find most frequent pair
      |
      v
    Merge pair into new token
      |
      v
    Repeat until vocab_size is reached
      |
      v
    Learned BPE vocabulary
      |
      +--------> encode(text)
      |
      +--------> decode(ids)




Example:


    tokenizer = BasicTokenizer()


    tokenizer.train(
        text="hello hello hello",
        vocab_size=270
    )


    ids = tokenizer.encode("hello")


    print(ids)


    print(tokenizer.decode(ids))
"""


from collections import Counter


import regex as re




# ==============================================================
# CONFIGURATION
# ==============================================================


# GPT-4 / cl100k_base-style pre-tokenization pattern.
#
# This separates:
# - contractions
# - words
# - numbers
# - punctuation
# - whitespace
#
# BPE will then operate independently inside each chunk.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)"""
    r"""|[^\r\n\p{L}\p{N}]?+\p{L}+"""
    r"""|\p{N}{1,3}"""
    r"""| ?[^\s\p{L}\p{N}]++[\r\n]*"""
    r"""|\s*[\r\n]"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)




# ==============================================================
# BPE HELPER FUNCTIONS
# ==============================================================


def get_stats(ids):
    """
    Count how many times every adjacent token pair occurs.


    Example:


        ids = [1, 2, 1, 2, 3]


    Adjacent pairs:


        (1, 2)
        (2, 1)
        (1, 2)
        (2, 3)


    Result:


        {
            (1, 2): 2,
            (2, 1): 1,
            (2, 3): 1
        }


    Parameters
    ----------
    ids : list[int]
        Token IDs.


    Returns
    -------
    dict[tuple[int, int], int]
        Mapping from token pair to occurrence count.
    """


    counts = {}


    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1


    return counts




def merge(ids, pair, idx):
    """
    Replace every non-overlapping occurrence of a token pair
    with a new token ID.


    Example:


        ids  = [5, 6, 7, 6, 7]
        pair = (6, 7)
        idx  = 99


    Result:


        [5, 99, 99]


    Parameters
    ----------
    ids : list[int]
        Existing token sequence.


    pair : tuple[int, int]
        Pair that should be merged.


    idx : int
        New token ID representing the pair.


    Returns
    -------
    list[int]
        Token sequence after merging.
    """


    new_ids = []


    i = 0


    while i < len(ids):


        # Check whether the next two tokens match the target pair.
        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            new_ids.append(idx)


            # Skip both original tokens.
            i += 2


        else:
            new_ids.append(ids[i])


            i += 1


    return new_ids




# ==============================================================
# BASIC TOKENIZER
# ==============================================================


class BasicTokenizer:
    """
    Simple byte-level BPE tokenizer.


    Unlike GPT-style tokenizers, this tokenizer DOES NOT perform
    regex pre-tokenization.


    BPE runs over the entire UTF-8 byte sequence.
    """


    def __init__(self):


        # Maps:
        #
        #     (token_a, token_b) -> new_token_id
        #
        # Example:
        #
        #     (104, 101) -> 256
        #
        # meaning byte/token 104 followed by 101 becomes token 256.
        self.merges = {}


        # Maps:
        #
        #     token_id -> bytes
        #
        # Example:
        #
        #     104 -> b'h'
        #     256 -> b'he'
        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }


    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------


    def train(
        self,
        text,
        vocab_size,
        verbose=False,
    ):
        """
        Learn BPE merges from training text.


        Parameters
        ----------
        text : str
            Training corpus.


        vocab_size : int
            Desired vocabulary size.


            The first 256 tokens are reserved for raw byte values,
            therefore vocab_size must be >= 256.


        verbose : bool
            Print each learned merge.
        """


        if vocab_size < 256:
            raise ValueError(
                "vocab_size must be at least 256"
            )


        # ------------------------------------------------------
        # STEP 1
        # Convert text to UTF-8 bytes.
        # ------------------------------------------------------


        tokens = list(
            text.encode("utf-8")
        )


        # ------------------------------------------------------
        # STEP 2
        # Initialize vocabulary with all possible byte values.
        # ------------------------------------------------------


        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }


        self.merges = {}


        # Number of additional tokens we want to learn.
        num_merges = vocab_size - 256


        # ------------------------------------------------------
        # STEP 3
        # Repeatedly find and merge most frequent pair.
        # ------------------------------------------------------


        for i in range(num_merges):


            stats = get_stats(tokens)


            # No pairs remain.
            if not stats:
                break


            # Most frequently occurring adjacent pair.
            pair = max(
                stats,
                key=stats.get,
            )


            count = stats[pair]


            # Optional:
            # Do not create tokens for pairs occurring only once.
            #
            # This is useful for a small educational tokenizer.
            if count < 2:
                break


            # New tokens begin after byte tokens 0-255.
            idx = 256 + i


            # Save merge rule.
            self.merges[pair] = idx


            # Build the byte representation of this new token.
            self.vocab[idx] = (
                self.vocab[pair[0]]
                + self.vocab[pair[1]]
            )


            # Actually apply the merge.
            tokens = merge(
                tokens,
                pair,
                idx,
            )


            if verbose:


                decoded = self.vocab[idx].decode(
                    "utf-8",
                    errors="replace",
                )


                print(
                    f"Merge {i + 1:02d}: "
                    f"{pair} -> {idx} | "
                    f"{repr(decoded)} | "
                    f"count={count}"
                )


    # ----------------------------------------------------------
    # ENCODE
    # ----------------------------------------------------------


    def encode(self, text):
        """
        Encode text into BPE token IDs.


        Important:
        We must apply merges according to the order in which they
        were learned during training.
        """


        # Start with raw UTF-8 byte IDs.
        tokens = list(
            text.encode("utf-8")
        )


        while len(tokens) >= 2:


            stats = get_stats(tokens)


            if not stats:
                break


            # Each learned merge has an ID:
            #
            # 256 = first merge
            # 257 = second merge
            # ...
            #
            # Therefore selecting the smallest merge ID gives us
            # the earliest applicable merge.
            pair = min(
                stats,
                key=lambda p: self.merges.get(
                    p,
                    float("inf"),
                ),
            )


            # None of the current pairs were learned.
            if pair not in self.merges:
                break


            idx = self.merges[pair]


            tokens = merge(
                tokens,
                pair,
                idx,
            )


        return tokens


    # ----------------------------------------------------------
    # DECODE
    # ----------------------------------------------------------


    def decode(self, ids):
        """
        Convert token IDs back into text.
        """


        try:


            raw_bytes = b"".join(
                self.vocab[idx]
                for idx in ids
            )


        except KeyError as error:


            raise ValueError(
                f"Unknown token ID: {error.args[0]}"
            )


        return raw_bytes.decode(
            "utf-8",
            errors="replace",
        )


    # ----------------------------------------------------------
    # DISPLAY VOCAB
    # ----------------------------------------------------------


    def print_learned_tokens(self):
        """
        Print only tokens created by BPE.


        Raw byte tokens 0-255 are omitted.
        """


        print("\nLearned BPE tokens:")


        for token_id in sorted(self.vocab):


            if token_id < 256:
                continue


            token_bytes = self.vocab[token_id]


            token_text = token_bytes.decode(
                "utf-8",
                errors="replace",
            )


            print(
                token_id,
                repr(token_text),
            )




# ==============================================================
# GPT-4 STYLE REGEX TOKENIZER
# ==============================================================


class RegexTokenizer:
    """
    Byte-level BPE tokenizer with GPT-4-style pre-tokenization.


    Main difference from BasicTokenizer:


        BasicTokenizer:


            text
              |
              v
            all UTF-8 bytes
              |
              v
            BPE


        RegexTokenizer:


            text
              |
              v
            regex split
              |
              +--> chunk 1 -> bytes -> BPE
              +--> chunk 2 -> bytes -> BPE
              +--> chunk 3 -> bytes -> BPE


    BPE never crosses chunk boundaries.
    """


    def __init__(
        self,
        pattern=GPT4_SPLIT_PATTERN,
    ):


        self.pattern = re.compile(pattern)


        self.merges = {}


        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }


    # ----------------------------------------------------------
    # PRE-TOKENIZATION
    # ----------------------------------------------------------


    def split_text(self, text):
        """
        Split text using GPT-4-style regex.
        """


        return re.findall(
            self.pattern,
            text,
        )


    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------


    def train(
        self,
        text,
        vocab_size,
        verbose=False,
    ):
        """
        Train BPE independently across regex chunks.
        """


        if vocab_size < 256:


            raise ValueError(
                "vocab_size must be at least 256"
            )


        self.merges = {}


        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }


        # ------------------------------------------------------
        # STEP 1
        # Regex pre-tokenization.
        # ------------------------------------------------------


        chunks = self.split_text(text)


        # ------------------------------------------------------
        # STEP 2
        # Convert every chunk independently into bytes.
        # ------------------------------------------------------


        token_chunks = [
            list(chunk.encode("utf-8"))
            for chunk in chunks
        ]


        # ------------------------------------------------------
        # STEP 3
        # Learn BPE merges.
        # ------------------------------------------------------


        for new_id in range(
            256,
            vocab_size,
        ):


            counts = Counter()


            # Count token pairs INSIDE each chunk.
            #
            # Importantly, we do not count a pair between
            # the last token of chunk 1 and first token of chunk 2.
            for tokens in token_chunks:


                counts.update(
                    zip(
                        tokens,
                        tokens[1:],
                    )
                )


            if not counts:
                break


            pair, count = counts.most_common(1)[0]


            if count < 2:
                break


            a, b = pair


            self.merges[pair] = new_id


            self.vocab[new_id] = (
                self.vocab[a]
                + self.vocab[b]
            )


            # --------------------------------------------------
            # Apply this merge independently to every chunk.
            # --------------------------------------------------


            for chunk_index, tokens in enumerate(
                token_chunks
            ):


                token_chunks[chunk_index] = merge(
                    tokens,
                    pair,
                    new_id,
                )


            if verbose:


                decoded = self.vocab[new_id].decode(
                    "utf-8",
                    errors="replace",
                )


                print(
                    f"Merge {new_id}: "
                    f"{pair} -> {new_id} | "
                    f"{repr(decoded)} | "
                    f"count={count}"
                )


    # ----------------------------------------------------------
    # ENCODE
    # ----------------------------------------------------------


    def encode(self, text):
        """
        Encode text using:


            regex split
                  +
            learned BPE merges
        """


        chunks = self.split_text(text)


        final_ids = []


        for chunk in chunks:


            tokens = list(
                chunk.encode("utf-8")
            )


            # Continue applying the earliest learned applicable
            # merge until no merge is possible.
            while len(tokens) >= 2:


                stats = get_stats(tokens)


                if not stats:
                    break


                pair = min(
                    stats,
                    key=lambda p: self.merges.get(
                        p,
                        float("inf"),
                    ),
                )


                if pair not in self.merges:
                    break


                tokens = merge(
                    tokens,
                    pair,
                    self.merges[pair],
                )


            final_ids.extend(tokens)


        return final_ids


    # ----------------------------------------------------------
    # DECODE
    # ----------------------------------------------------------


    def decode(self, ids):
        """
        Decode BPE token IDs into text.
        """


        try:


            raw_bytes = b"".join(
                self.vocab[idx]
                for idx in ids
            )


        except KeyError as error:


            raise ValueError(
                f"Unknown token ID: {error.args[0]}"
            )


        return raw_bytes.decode(
            "utf-8",
            errors="replace",
        )


    # ----------------------------------------------------------
    # DISPLAY VOCAB
    # ----------------------------------------------------------


    def print_learned_tokens(self):


        print("\nLearned Regex BPE tokens:")


        for token_id in sorted(self.vocab):


            if token_id < 256:
                continue


            token_text = self.vocab[token_id].decode(
                "utf-8",
                errors="replace",
            )


            print(
                token_id,
                repr(token_text),
            )




# ==============================================================
# DEMO TRAINING DATA
# ==============================================================


SAMPLE_TEXT = """
Taylor Swift is a singer and songwriter.
Taylor Swift writes songs about love, friendship, and life.
Her music includes pop, country, and indie influences.


Taylor Swift writes music.
Taylor Swift writes songs.
Taylor Swift writes lyrics.
"""




# ==============================================================
# DEMO 1
# UTF-8
# ==============================================================


def demo_utf8():


    print("\n")
    print("=" * 70)
    print("1. UTF-8 BYTE REPRESENTATION")
    print("=" * 70)


    text = "안녕하세요 (hello in Korean!)"


    encoded = text.encode("utf-8")


    print("Original text:")
    print(text)


    print("\nUTF-8 bytes:")
    print(encoded)


    print("\nByte IDs:")
    print(list(encoded))


    print(
        "\nCharacters:",
        len(text),
    )


    print(
        "UTF-8 bytes:",
        len(encoded),
    )




# ==============================================================
# DEMO 2
# PAIR STATISTICS
# ==============================================================


def demo_pair_statistics():


    print("\n")
    print("=" * 70)
    print("2. COUNTING ADJACENT TOKEN PAIRS")
    print("=" * 70)


    text = "hello hello hello"


    tokens = list(
        text.encode("utf-8")
    )


    print("Tokens:")
    print(tokens)


    stats = get_stats(tokens)


    print("\nPair counts:")


    for pair, count in sorted(
        stats.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            pair,
            "->",
            count,
        )


    top_pair = max(
        stats,
        key=stats.get,
    )


    print(
        "\nMost frequent pair:",
        top_pair,
    )


    print(
        "Count:",
        stats[top_pair],
    )




# ==============================================================
# DEMO 3
# MANUAL MERGE
# ==============================================================


def demo_manual_merge():


    print("\n")
    print("=" * 70)
    print("3. MANUAL BPE MERGE")
    print("=" * 70)


    ids = [
        5,
        6,
        7,
        6,
        7,
        9,
    ]


    pair = (6, 7)


    result = merge(
        ids,
        pair,
        99,
    )


    print(
        "Original:",
        ids,
    )


    print(
        "Merge pair:",
        pair,
    )


    print(
        "New token ID:",
        99,
    )


    print(
        "Result:",
        result,
    )




# ==============================================================
# DEMO 4
# BASIC TOKENIZER
# ==============================================================


def demo_basic_tokenizer():


    print("\n")
    print("=" * 70)
    print("4. BASIC BYTE-LEVEL BPE TOKENIZER")
    print("=" * 70)


    tokenizer = BasicTokenizer()


    tokenizer.train(
        SAMPLE_TEXT,
        vocab_size=300,
        verbose=True,
    )


    tokenizer.print_learned_tokens()


    sample = "Taylor Swift writes songs."


    ids = tokenizer.encode(sample)


    decoded = tokenizer.decode(ids)


    print("\nOriginal:")
    print(sample)


    print("\nToken IDs:")
    print(ids)


    print(
        "\nNumber of UTF-8 bytes:",
        len(sample.encode("utf-8")),
    )


    print(
        "Number of BPE tokens:",
        len(ids),
    )


    print("\nDecoded:")
    print(decoded)


    print(
        "\nRound-trip successful:",
        decoded == sample,
    )


    assert decoded == sample




# ==============================================================
# DEMO 5
# REGEX PRE-TOKENIZATION
# ==============================================================


def demo_regex_split():


    print("\n")
    print("=" * 70)
    print("5. GPT-4-STYLE REGEX PRE-TOKENIZATION")
    print("=" * 70)


    tokenizer = RegexTokenizer()


    examples = [
        "Hello world how are you",
        "I'm testing GPT-style tokenization.",
        "hello123 world456!!!",
        "    hello world!!!",
    ]


    for text in examples:


        print("\nInput:")
        print(repr(text))


        print("Chunks:")


        print(
            tokenizer.split_text(text)
        )




# ==============================================================
# DEMO 6
# REGEX TOKENIZER
# ==============================================================


def demo_regex_tokenizer():


    print("\n")
    print("=" * 70)
    print("6. REGEX + BPE TOKENIZER")
    print("=" * 70)


    tokenizer = RegexTokenizer()


    tokenizer.train(
        SAMPLE_TEXT,
        vocab_size=300,
        verbose=True,
    )


    tokenizer.print_learned_tokens()


    sample = "Taylor Swift writes songs."


    ids = tokenizer.encode(sample)


    decoded = tokenizer.decode(ids)


    print("\nOriginal:")
    print(sample)


    print("\nToken IDs:")
    print(ids)


    print("\nDecoded:")
    print(decoded)


    print(
        "\nRound-trip successful:",
        decoded == sample,
    )


    assert decoded == sample




# ==============================================================
# DEMO 7
# BASIC VS REGEX TOKENIZER
# ==============================================================


def demo_basic_vs_regex():


    print("\n")
    print("=" * 70)
    print("7. BASIC BPE VS REGEX BPE")
    print("=" * 70)


    basic = BasicTokenizer()


    regex_tokenizer = RegexTokenizer()


    basic.train(
        SAMPLE_TEXT,
        vocab_size=300,
    )


    regex_tokenizer.train(
        SAMPLE_TEXT,
        vocab_size=300,
    )


    print("\n--- BASIC TOKENIZER MERGES ---")


    basic.print_learned_tokens()


    print("\n--- REGEX TOKENIZER MERGES ---")


    regex_tokenizer.print_learned_tokens()


    test_text = "Taylor Swift writes songs."


    basic_ids = basic.encode(test_text)


    regex_ids = regex_tokenizer.encode(
        test_text
    )


    print("\nText:")
    print(test_text)


    print("\nBasic tokenizer IDs:")
    print(basic_ids)


    print("\nRegex tokenizer IDs:")
    print(regex_ids)


    print(
        "\nBasic token count:",
        len(basic_ids),
    )


    print(
        "Regex token count:",
        len(regex_ids),
    )




# ==============================================================
# DEMO 8
# UNICODE ROUND-TRIP
# ==============================================================


def demo_unicode():


    print("\n")
    print("=" * 70)
    print("8. UNICODE ROUND-TRIP")
    print("=" * 70)


    training_text = """
    Hello hello hello.
    안녕하세요 안녕하세요.
    नमस्ते नमस्ते.
    Hello world.
    """


    tokenizer = BasicTokenizer()


    tokenizer.train(
        training_text,
        vocab_size=300,
    )


    tests = [
        "hello",
        "안녕하세요",
        "नमस्ते",
        "hello 👋 world",
        "",
    ]


    for text in tests:


        ids = tokenizer.encode(text)


        decoded = tokenizer.decode(ids)


        print("\nInput:")
        print(repr(text))


        print("IDs:")
        print(ids)


        print("Decoded:")
        print(repr(decoded))


        print(
            "Correct:",
            text == decoded,
        )


        assert text == decoded




# ==============================================================
# DEMO 9
# TIKTOKEN COMPARISON
# ==============================================================


def demo_tiktoken():


    print("\n")
    print("=" * 70)
    print("9. COMPARISON WITH TIKTOKEN")
    print("=" * 70)


    try:
        import tiktoken


    except ImportError:


        print(
            "tiktoken is not installed."
        )


        print(
            "Install it using:"
        )


        print(
            "pip install tiktoken"
        )


        return


    text = "    hello world!!!"


    # ----------------------------------------------------------
    # GPT-2 tokenizer
    # ----------------------------------------------------------


    gpt2 = tiktoken.get_encoding(
        "gpt2"
    )


    gpt2_ids = gpt2.encode(text)


    # ----------------------------------------------------------
    # cl100k_base
    # Used by several GPT-family models.
    # ----------------------------------------------------------


    cl100k = tiktoken.get_encoding(
        "cl100k_base"
    )


    cl100k_ids = cl100k.encode(text)


    print("\nInput:")
    print(repr(text))


    print("\nGPT-2:")
    print(gpt2_ids)


    print(
        "Token count:",
        len(gpt2_ids),
    )


    print("\ncl100k_base:")
    print(cl100k_ids)


    print(
        "Token count:",
        len(cl100k_ids),
    )


    # ----------------------------------------------------------
    # Unicode example
    # ----------------------------------------------------------


    unicode_text = (
        "안녕하세요 👋 "
        "(hello in Korean!)"
    )


    encoded = cl100k.encode(
        unicode_text
    )


    decoded = cl100k.decode(
        encoded
    )


    print("\nUnicode input:")
    print(unicode_text)


    print("\nToken IDs:")
    print(encoded)


    print("\nDecoded:")
    print(decoded)


    print(
        "\nRound-trip successful:",
        decoded == unicode_text,
    )




# ==============================================================
# DEMO 10
# COMPRESSION RATIO
# ==============================================================


def demo_compression():


    print("\n")
    print("=" * 70)
    print("10. BPE COMPRESSION")
    print("=" * 70)


    tokenizer = BasicTokenizer()


    tokenizer.train(
        SAMPLE_TEXT,
        vocab_size=300,
    )


    original_bytes = list(
        SAMPLE_TEXT.encode("utf-8")
    )


    encoded_ids = tokenizer.encode(
        SAMPLE_TEXT
    )


    print(
        "Original UTF-8 byte count:",
        len(original_bytes),
    )


    print(
        "BPE token count:",
        len(encoded_ids),
    )


    if encoded_ids:


        ratio = (
            len(original_bytes)
            / len(encoded_ids)
        )


        print(
            f"Compression ratio: {ratio:.2f}X"
        )




# ==============================================================
# MAIN
# ==============================================================


def main():
    """
    Run all tokenizer demonstrations.
    """


    demo_utf8()


    demo_pair_statistics()


    demo_manual_merge()


    demo_basic_tokenizer()


    demo_regex_split()


    demo_regex_tokenizer()


    demo_basic_vs_regex()


    demo_unicode()


    demo_compression()


    demo_tiktoken()




# ==============================================================
# ENTRY POINT
# ==============================================================


if __name__ == "__main__":
    main()



