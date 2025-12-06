"""NLP dataset classes for text processing and sequence modeling."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional, Union

import mlx.core as mx

Array = mx.array


class TextDataset:
    """Dataset for text classification and other text-based tasks.

    Handles raw text data with optional tokenization and encoding.

    Args:
        texts: List of text strings or path to a text file (one text per line).
        labels: Optional list of labels (integers or strings).
        tokenizer: Optional tokenizer function. If None, uses simple whitespace split.
        vocab: Optional vocabulary mapping (word -> index). If None, builds from data.
        max_length: Maximum sequence length. Longer sequences are truncated.
        pad_token: Token to use for padding. Default is "<PAD>".
        unk_token: Token to use for unknown words. Default is "<UNK>".
        label_map: Optional mapping from label strings to integers.

    Example:
        >>> texts = ["Hello world", "How are you"]
        >>> labels = [0, 1]
        >>> dataset = TextDataset(texts, labels, max_length=10)
        >>> sample = dataset[0]
        >>> print(sample["x"].shape)  # (10,) padded sequence
        >>> print(sample["y"])  # 0
    """

    def __init__(
        self,
        texts: Union[List[str], str],
        labels: Optional[Union[List[Any], str]] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        vocab: Optional[Dict[str, int]] = None,
        max_length: int = 128,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        label_map: Optional[Dict[Any, int]] = None
    ) -> None:
        # Load texts from file if path provided
        if isinstance(texts, str):
            if os.path.isfile(texts):
                with open(texts, "r", encoding="utf-8") as f:
                    texts = [line.strip() for line in f if line.strip()]
            else:
                raise ValueError(f"Text file not found: {texts}")

        self.texts = texts
        self.max_length = max_length
        self.pad_token = pad_token
        self.unk_token = unk_token

        # Set up tokenizer
        self.tokenizer = tokenizer or self._default_tokenizer

        # Build or use provided vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = self._build_vocab(texts)

        # Ensure special tokens are in vocab
        if pad_token not in self.vocab:
            self.vocab[pad_token] = len(self.vocab)
        if unk_token not in self.vocab:
            self.vocab[unk_token] = len(self.vocab)

        self.pad_idx = self.vocab[pad_token]
        self.unk_idx = self.vocab[unk_token]

        # Handle labels
        self.labels: Optional[List[int]] = None
        self.label_map = label_map

        if labels is not None:
            if isinstance(labels, str) and os.path.isfile(labels):
                with open(labels, "r", encoding="utf-8") as f:
                    labels = [line.strip() for line in f if line.strip()]

            if label_map is not None:
                self.labels = [label_map[label] for label in labels]
            elif all(isinstance(label, int) for label in labels):
                self.labels = list(labels)
            else:
                # Build label map from unique labels
                unique_labels = sorted(set(labels))
                self.label_map = {label: i for i, label in enumerate(unique_labels)}
                self.labels = [self.label_map[label] for label in labels]

    def _default_tokenizer(self, text: str) -> List[str]:
        """Simple whitespace tokenizer with lowercasing."""
        return text.lower().split()

    def _build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """Build vocabulary from texts."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            for token in self.tokenizer(text):
                word_counts[token] = word_counts.get(token, 0) + 1

        # Sort by frequency (most common first)
        sorted_words = sorted(word_counts.keys(), key=lambda x: -word_counts[x])
        vocab = {word: idx for idx, word in enumerate(sorted_words)}
        return vocab

    def _encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        tokens = self.tokenizer(text)
        indices = [self.vocab.get(t, self.unk_idx) for t in tokens]

        # Truncate if too long
        if len(indices) > self.max_length:
            indices = indices[: self.max_length]

        # Pad if too short
        while len(indices) < self.max_length:
            indices.append(self.pad_idx)

        return indices

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        encoded = self._encode(self.texts[idx])
        result: Dict[str, Array] = {"x": mx.array(encoded)}

        if self.labels is not None:
            result["y"] = mx.array(self.labels[idx])

        return result

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)

    @property
    def num_classes(self) -> Optional[int]:
        """Return number of classes if labels are provided."""
        if self.label_map is not None:
            return len(self.label_map)
        if self.labels is not None:
            return max(self.labels) + 1
        return None

    def save_vocab(self, path: str) -> None:
        """Save vocabulary to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load_vocab(cls, path: str) -> Dict[str, int]:
        """Load vocabulary from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


class SequenceDataset:
    """Dataset for sequence-to-sequence tasks.

    Handles paired sequences like translation, summarization, etc.

    Args:
        source_texts: List of source text strings.
        target_texts: List of target text strings.
        source_tokenizer: Tokenizer for source texts.
        target_tokenizer: Tokenizer for target texts.
        source_vocab: Vocabulary for source language.
        target_vocab: Vocabulary for target language.
        max_source_length: Maximum source sequence length.
        max_target_length: Maximum target sequence length.
        pad_token: Padding token.
        unk_token: Unknown token.
        bos_token: Beginning of sequence token.
        eos_token: End of sequence token.

    Example:
        >>> source = ["Hello", "How are you"]
        >>> target = ["Bonjour", "Comment allez-vous"]
        >>> dataset = SequenceDataset(source, target, max_source_length=10, max_target_length=15)
    """

    def __init__(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_tokenizer: Optional[Callable[[str], List[str]]] = None,
        target_tokenizer: Optional[Callable[[str], List[str]]] = None,
        source_vocab: Optional[Dict[str, int]] = None,
        target_vocab: Optional[Dict[str, int]] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>"
    ) -> None:
        if len(source_texts) != len(target_texts):
            raise ValueError(
                f"Source and target must have same length: "
                f"{len(source_texts)} vs {len(target_texts)}"
            )

        self.source_texts = source_texts
        self.target_texts = target_texts
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        # Special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Tokenizers
        self.source_tokenizer = source_tokenizer or (lambda x: x.lower().split())
        self.target_tokenizer = target_tokenizer or (lambda x: x.lower().split())

        # Build vocabularies
        self.source_vocab = source_vocab or self._build_vocab(
            source_texts, self.source_tokenizer
        )
        self.target_vocab = target_vocab or self._build_vocab(
            target_texts, self.target_tokenizer
        )

        # Add special tokens to vocabularies
        for token in [pad_token, unk_token, bos_token, eos_token]:
            if token not in self.source_vocab:
                self.source_vocab[token] = len(self.source_vocab)
            if token not in self.target_vocab:
                self.target_vocab[token] = len(self.target_vocab)

        # Cache special token indices
        self.source_pad_idx = self.source_vocab[pad_token]
        self.source_unk_idx = self.source_vocab[unk_token]
        self.target_pad_idx = self.target_vocab[pad_token]
        self.target_unk_idx = self.target_vocab[unk_token]
        self.target_bos_idx = self.target_vocab[bos_token]
        self.target_eos_idx = self.target_vocab[eos_token]

    def _build_vocab(
        self,
        texts: List[str],
        tokenizer: Callable[[str], List[str]]
    ) -> Dict[str, int]:
        """Build vocabulary from texts."""
        word_counts: Dict[str, int] = {}
        for text in texts:
            for token in tokenizer(text):
                word_counts[token] = word_counts.get(token, 0) + 1

        sorted_words = sorted(word_counts.keys(), key=lambda x: -word_counts[x])
        return {word: idx for idx, word in enumerate(sorted_words)}

    def _encode_source(self, text: str) -> List[int]:
        """Encode source text."""
        tokens = self.source_tokenizer(text)
        indices = [self.source_vocab.get(t, self.source_unk_idx) for t in tokens]

        if len(indices) > self.max_source_length:
            indices = indices[: self.max_source_length]

        while len(indices) < self.max_source_length:
            indices.append(self.source_pad_idx)

        return indices

    def _encode_target(self, text: str) -> List[int]:
        """Encode target text with BOS and EOS tokens."""
        tokens = self.target_tokenizer(text)
        indices = [self.target_bos_idx]
        indices.extend(self.target_vocab.get(t, self.target_unk_idx) for t in tokens)
        indices.append(self.target_eos_idx)

        if len(indices) > self.max_target_length:
            indices = indices[: self.max_target_length - 1] + [self.target_eos_idx]

        while len(indices) < self.max_target_length:
            indices.append(self.target_pad_idx)

        return indices

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        source_encoded = self._encode_source(self.source_texts[idx])
        target_encoded = self._encode_target(self.target_texts[idx])

        return {
            "source": mx.array(source_encoded),
            "target": mx.array(target_encoded),
            # For teacher forcing: input is target[:-1], output is target[1:]
            "target_input": mx.array(target_encoded[:-1]),
            "target_output": mx.array(target_encoded[1:]),
        }

    @property
    def source_vocab_size(self) -> int:
        return len(self.source_vocab)

    @property
    def target_vocab_size(self) -> int:
        return len(self.target_vocab)


class TokenizedDataset:
    """Dataset for pre-tokenized data (e.g., from HuggingFace tokenizers).

    Accepts already tokenized data as lists of token IDs.

    Args:
        input_ids: List of token ID sequences.
        attention_mask: Optional list of attention masks.
        labels: Optional list of labels.
        token_type_ids: Optional list of token type IDs (for BERT-style models).

    Example:
        >>> # From HuggingFace tokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
        >>> dataset = TokenizedDataset(
        ...     input_ids=encoded["input_ids"].tolist(),
        ...     attention_mask=encoded["attention_mask"].tolist(),
        ...     labels=labels
        ... )
    """

    def __init__(
        self,
        input_ids: List[List[int]],
        attention_mask: Optional[List[List[int]]] = None,
        labels: Optional[List[int]] = None,
        token_type_ids: Optional[List[List[int]]] = None
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.token_type_ids = token_type_ids

        # Validate lengths
        n = len(input_ids)
        if attention_mask is not None and len(attention_mask) != n:
            raise ValueError("attention_mask length must match input_ids")
        if labels is not None and len(labels) != n:
            raise ValueError("labels length must match input_ids")
        if token_type_ids is not None and len(token_type_ids) != n:
            raise ValueError("token_type_ids length must match input_ids")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        result: Dict[str, Array] = {
            "input_ids": mx.array(self.input_ids[idx]),
            "x": mx.array(self.input_ids[idx]),  # Alias for compatibility
        }

        if self.attention_mask is not None:
            result["attention_mask"] = mx.array(self.attention_mask[idx])

        if self.token_type_ids is not None:
            result["token_type_ids"] = mx.array(self.token_type_ids[idx])

        if self.labels is not None:
            result["y"] = mx.array(self.labels[idx])
            result["labels"] = mx.array(self.labels[idx])

        return result


class LanguageModelDataset:
    """Dataset for language modeling (next token prediction).

    Creates sequences for autoregressive language modeling where
    the target is the input shifted by one position.

    Args:
        text: Text string or path to text file.
        tokenizer: Tokenizer function.
        vocab: Vocabulary mapping.
        seq_length: Sequence length for each sample.
        stride: Stride between consecutive samples. If None, uses seq_length (no overlap).

    Example:
        >>> dataset = LanguageModelDataset(
        ...     text="path/to/corpus.txt",
        ...     seq_length=128,
        ...     stride=64  # 50% overlap
        ... )
    """

    def __init__(
        self,
        text: str,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        vocab: Optional[Dict[str, int]] = None,
        seq_length: int = 128,
        stride: Optional[int] = None,
        unk_token: str = "<UNK>"
    ) -> None:
        # Load text from file if path
        if os.path.isfile(text):
            with open(text, "r", encoding="utf-8") as f:
                text = f.read()

        self.seq_length = seq_length
        self.stride = stride or seq_length
        self.unk_token = unk_token

        # Tokenizer
        self.tokenizer = tokenizer or (lambda x: list(x))  # Character-level by default

        # Tokenize entire text
        tokens = self.tokenizer(text)

        # Build or use vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            unique_tokens = sorted(set(tokens))
            self.vocab = {t: i for i, t in enumerate(unique_tokens)}

        if unk_token not in self.vocab:
            self.vocab[unk_token] = len(self.vocab)

        self.unk_idx = self.vocab[unk_token]

        # Encode all tokens
        self.token_ids = [self.vocab.get(t, self.unk_idx) for t in tokens]

        # Calculate number of samples
        if len(self.token_ids) <= seq_length:
            self._num_samples = 1
        else:
            self._num_samples = (len(self.token_ids) - seq_length) // self.stride + 1

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> Dict[str, Array]:
        start = idx * self.stride
        end = start + self.seq_length

        # Ensure we don't go past the end
        if end >= len(self.token_ids):
            end = len(self.token_ids) - 1
            start = end - self.seq_length

        input_ids = self.token_ids[start:end]
        target_ids = self.token_ids[start + 1: end + 1]

        return {
            "x": mx.array(input_ids),
            "y": mx.array(target_ids),
            "input_ids": mx.array(input_ids),
            "labels": mx.array(target_ids),
        }

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
