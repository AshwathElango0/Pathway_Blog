import unicodedata
import pathway as pw
from pathway.optional_import import optional_imports

@pw.udf
def null_splitter(txt: str) -> list[tuple[str, dict]]:
    """A splitter which returns its argument as one long text ith null metadata.

    Args:
        txt: text to be split

    Returns:
        list of pairs: chunk text and metadata.

    The null splitter always return a list of length one containing the full text and empty metadata.
    """
    return [(txt, {})]


def _normalize_unicode(text: str):
    """Normalize Unicode characters."""
    return unicodedata.normalize("NFKC", text)


class BaseTokenSplitter(pw.UDF):
    """Base class for token-based text splitting."""

    CHARS_PER_TOKEN = 3
    PUNCTUATION = [".", "?", "!", "\n"]

    def __init__(self, encoding_name: str = "cl100k_base"):
        with optional_imports("xpack-llm"):
            import tiktoken  # noqa:F401

        super().__init__()
        self.encoding_name = encoding_name
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def _tokenize(self, text: str):
        """Tokenize and normalize the text."""
        text = _normalize_unicode(text)
        return self.tokenizer.encode_ordinary(text)

    def __call__(self, text: pw.ColumnExpression, **kwargs) -> pw.ColumnExpression:
        """Split given strings into smaller chunks.

        Args:
            text (ColumnExpression[str]): Column with texts to be split.
            **kwargs: override for defaults set in the constructor.
        """
        return super().__call__(text, **kwargs)


class DefaultTokenCountSplitter(BaseTokenSplitter):
    """
    Splits text into chunks based on min and max token limits.
    """

    def __init__(self, min_tokens: int = 50, max_tokens: int = 500, encoding_name: str = "cl100k_base"):
        super().__init__(encoding_name)
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
        tokens = self._tokenize(txt)
        output = []
        i = 0

        while i < len(tokens):
            chunk_tokens = tokens[i : i + self.max_tokens]
            chunk = self.tokenizer.decode(chunk_tokens)
            last_punctuation = max([chunk.rfind(p) for p in self.PUNCTUATION], default=-1)

            if last_punctuation != -1 and last_punctuation > self.CHARS_PER_TOKEN * self.min_tokens:
                chunk = chunk[: last_punctuation + 1]

            i += len(self._tokenize(chunk))
            output.append((chunk, {}))

        return output
class SlidingWindowSplitter(BaseTokenSplitter):
    """
    Splits text into overlapping chunks with a sliding window.
    """

    def __init__(self, max_tokens: int = 500, overlap_tokens: int = 20, encoding_name: str = "cl100k_base"):
        super().__init__(encoding_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
        tokens = self._tokenize(txt)
        output = []
        i = 0

        while i < len(tokens):
            end = min(i + self.max_tokens, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk = self.tokenizer.decode(chunk_tokens)
            output.append((chunk, {}))
            i += self.max_tokens - self.overlap_tokens

        return output

class SmallToBigSplitter(BaseTokenSplitter):
    """
    Splits text into small and large chunks for small-to-big chunking.
    """

    def __init__(self, small_chunk_size: int = 175, large_chunk_size: int = 512, overlap_tokens: int = 20, encoding_name: str = "cl100k_base"):
        super().__init__(encoding_name)
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.overlap_tokens = overlap_tokens

    def __wrapped__(self, txt: str) -> list[tuple[str, dict]]:
        tokens = self._tokenize(txt)
        output = []

        # Small chunks
        i = 0
        while i < len(tokens):
            end = min(i + self.small_chunk_size, len(tokens))
            small_chunk_tokens = tokens[i:end]
            small_chunk = self.tokenizer.decode(small_chunk_tokens)
            output.append((small_chunk, {"chunk_size": "small"}))
            i += self.small_chunk_size - self.overlap_tokens

        # Large chunks
        i = 0
        while i < len(tokens):
            end = min(i + self.large_chunk_size, len(tokens))
            large_chunk_tokens = tokens[i:end]
            large_chunk = self.tokenizer.decode(large_chunk_tokens)
            output.append((large_chunk, {"chunk_size": "large"}))
            i += self.large_chunk_size - self.overlap_tokens

        return output
