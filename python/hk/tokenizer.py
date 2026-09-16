"""
HK Tokenization & AutoTokenizer
Standardized in-file tokenizer metadata (tokenizer.tokens, tokenizer.scores, tokenizer.merges),
Jinja2 chat templates, and AutoTokenizer in-file reconstruction.
"""

import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .constants import TokenType, PreTokenizerType
from .native import is_native_available, NativeHKTokenizer

try:
    import jinja2
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% endif %}"
)


def _render_chat_template_fallback(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    template_str: Optional[str] = None,
) -> str:
    """Lightweight fallback renderer for chat templates when Jinja2 is not installed."""
    out = []
    use_chatml = True
    if template_str and "<|im_start|>" not in template_str and "[INST]" in template_str:
        use_chatml = False

    if use_chatml:
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
    else:
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("system", "user"):
                out.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                out.append(f" {content} ")
        if add_generation_prompt:
            out.append(" [INST] ")

    return "".join(out)


# ---------------------------------------------------------------------------
# Pre-Tokenizer Regexes & Normalization
# ---------------------------------------------------------------------------

PRE_TOKENIZER_PATTERNS = {
    PreTokenizerType.DEFAULT: re.compile(r"\s+|[^\s\w]+|\w+"),
    PreTokenizerType.LLAMA3: re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\s]?[a-zA-Z]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
    PreTokenizerType.QWEN2: re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\s]?[a-zA-Z]+|\d+| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
    PreTokenizerType.DEEPSEEK_LLM: re.compile(r"[\r\n]+|[^\r\n\w\s]?[a-zA-Z]+|\d+| ?[^\s\w]+[\r\n]*|\s+(?!\S)|\s+"),
    PreTokenizerType.GPT2: re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\s]?[a-zA-Z]+|\d{1,3}| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
    PreTokenizerType.TEKKEN: re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w\s]?[a-zA-Z]+|\d+| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"),
    PreTokenizerType.BERT: re.compile(r"\s+|[^\s\w]+|\w+"),
}


def parse_sentencepiece_model(model_source: Union[bytes, str, Path]) -> Tuple[List[str], List[float], List[int]]:
    """
    Parses a SentencePiece binary .model file without requiring external protobuf or sentencepiece dependencies.
    Decodes the Protocol Buffer wire format (ModelProto) to extract tokens, scores, and token types.
    """
    if isinstance(model_source, (str, Path)):
        with open(model_source, "rb") as f:
            data = f.read()
    else:
        data = bytes(model_source)

    pos = 0
    total = len(data)

    def read_varint() -> int:
        nonlocal pos
        res = 0
        shift = 0
        while pos < total:
            b = data[pos]
            pos += 1
            res |= (b & 0x7F) << shift
            if (b & 0x80) == 0:
                break
            shift += 7
        return res

    tokens: List[str] = []
    scores: List[float] = []
    token_types: List[int] = []

    while pos < total:
        tag_wire = read_varint()
        field_num = tag_wire >> 3
        wire_type = tag_wire & 0x07

        if wire_type == 0:
            _ = read_varint()
        elif wire_type == 1:
            pos += 8
        elif wire_type == 2:
            length = read_varint()
            chunk_end = pos + length
            if field_num == 3:  # repeated SentencePiece pieces = 3 in ModelProto
                p_text = ""
                p_score = 0.0
                p_type = int(TokenType.NORMAL)
                while pos < chunk_end:
                    sub_tag = read_varint()
                    sub_field = sub_tag >> 3
                    sub_wire = sub_tag & 0x07
                    if sub_wire == 0:
                        v = read_varint()
                        if sub_field == 3:  # piece type
                            p_type = v
                    elif sub_wire == 1:
                        pos += 8
                    elif sub_wire == 2:
                        sub_len = read_varint()
                        sub_bytes = data[pos : pos + sub_len]
                        pos += sub_len
                        if sub_field == 1:  # piece text
                            try:
                                p_text = sub_bytes.decode("utf-8", errors="replace")
                            except Exception:
                                p_text = str(sub_bytes)
                    elif sub_wire == 5:
                        if sub_field == 2 and pos + 4 <= chunk_end:  # piece score
                            p_score = struct.unpack("<f", data[pos : pos + 4])[0]
                        pos += 4
                    else:
                        break
                tokens.append(p_text)
                scores.append(p_score)
                token_types.append(p_type)
                pos = chunk_end
            else:
                pos = chunk_end
        elif wire_type == 5:
            pos += 4
        else:
            break

    return tokens, scores, token_types


def parse_tekken_json(source: Union[dict, str, Path]) -> Tuple[List[str], List[float], List[Tuple[str, str]]]:
    """
    Parses a Mistral Tekkenizer JSON file (tekken.json).
    Extracts vocabulary tokens, ranks/scores, and BPE merges.
    """
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = source

    tokens: List[str] = []
    scores: List[float] = []
    merges: List[Tuple[str, str]] = []

    vocab_data = data.get("vocab", [])
    if isinstance(vocab_data, list):
        indexed_items = []
        for i, item in enumerate(vocab_data):
            if isinstance(item, dict):
                tok_str = item.get("token") or item.get("token_str") or item.get("piece") or str(i)
                rank = item.get("rank", i)
                score = float(item.get("score", -float(rank)))
                indexed_items.append((rank, tok_str, score))
            elif isinstance(item, str):
                indexed_items.append((i, item, -float(i)))
        indexed_items.sort(key=lambda x: x[0])
        tokens = [x[1] for x in indexed_items]
        scores = [x[2] for x in indexed_items]
    elif isinstance(vocab_data, dict):
        sorted_pairs = sorted(vocab_data.items(), key=lambda x: x[1])
        tokens = [k for k, _ in sorted_pairs]
        scores = [-float(v) for _, v in sorted_pairs]

    raw_merges = data.get("merges", [])
    for m in raw_merges:
        if isinstance(m, str):
            parts = m.strip().split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append((str(m[0]), str(m[1])))

    return tokens, scores, merges


class HKTokenizer:
    """Fast, portable subword/byte-level BPE tokenizer with in-file metadata persistence."""

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        tokens: Optional[List[str]] = None,
        scores: Optional[List[float]] = None,
        token_types: Optional[List[int]] = None,
        merges: Optional[List[Union[str, Tuple[str, str]]]] = None,
        pre_tokenizer: Optional[Union[str, PreTokenizerType]] = None,
        chat_template: Optional[str] = None,
        chat_templates: Optional[Dict[str, str]] = None,
        huggingface_json: Optional[str] = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.chat_template = chat_template or DEFAULT_CHAT_TEMPLATE
        self.chat_templates = dict(chat_templates) if chat_templates else ({"default": self.chat_template} if self.chat_template else {})
        self.huggingface_json = huggingface_json

        # Initialize vocab
        if tokens is not None:
            self.tokens = list(tokens)
            self.vocab = {tok: idx for idx, tok in enumerate(self.tokens)}
        elif vocab is not None:
            if isinstance(vocab, (list, tuple)):
                self.tokens = list(vocab)
                self.vocab = {tok: idx for idx, tok in enumerate(self.tokens)}
            else:
                self.vocab = dict(vocab)
                max_id = max(self.vocab.values()) if self.vocab else -1
                tok_list = [""] * (max_id + 1)
                for k, v in self.vocab.items():
                    if 0 <= v < len(tok_list):
                        tok_list[v] = k
                self.tokens = tok_list
        else:
            self.vocab = {
                self.pad_token: 0,
                self.unk_token: 1,
                self.bos_token: 2,
                self.eos_token: 3,
            }
            for i in range(256):
                ch = chr(i)
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)
            self.tokens = [k for k, _ in sorted(self.vocab.items(), key=lambda x: x[1])]

        self.scores = list(scores) if scores is not None else [0.0] * len(self.tokens)
        self.token_types = list(token_types) if token_types is not None else [int(TokenType.NORMAL)] * len(self.tokens)
        self.pre_tokenizer = str(pre_tokenizer) if pre_tokenizer is not None else None

        # Standardize merges
        self.merges: List[Tuple[str, str]] = []
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        if merges is not None:
            for rank, item in enumerate(merges):
                if isinstance(item, str):
                    parts = item.strip().split()
                    if len(parts) == 2:
                        pair = (parts[0], parts[1])
                        self.merges.append(pair)
                        self.bpe_ranks[pair] = rank
                elif isinstance(item, (list, tuple)) and len(item) == 2:
                    pair = (str(item[0]), str(item[1]))
                    self.merges.append(pair)
                    self.bpe_ranks[pair] = rank

        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab.get(self.pad_token, 0)
        self.unk_token_id = self.vocab.get(self.unk_token, 1)
        self.bos_token_id = self.vocab.get(self.bos_token, 2)
        self.eos_token_id = self.vocab.get(self.eos_token, 3)
        self._bpe_cache: Dict[str, List[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _bpe(self, token_word: str) -> List[str]:
        """Applies Byte-Pair Encoding merges to a single word/string."""
        if not self.bpe_ranks:
            return list(token_word)

        word = list(token_word)
        if len(word) <= 1:
            return word

        while True:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            min_pair = None
            min_rank = float("inf")
            for p in pairs:
                rank = self.bpe_ranks.get(p, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = p

            if min_pair is None or min_rank == float("inf"):
                break

            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            if len(word) <= 1:
                break

        return word

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if getattr(self, "_native_tok", None) is not None:
            try:
                if not text:
                    res = []
                    if add_special_tokens and self.bos_token_id is not None:
                        res.append(self.bos_token_id)
                    if add_special_tokens and self.eos_token_id is not None:
                        res.append(self.eos_token_id)
                    return res
                return self._native_tok.encode(text, add_bos=add_special_tokens, add_eos=add_special_tokens)
            except Exception:
                pass

        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)

        if not text:
            if add_special_tokens:
                tokens.append(self.eos_token_id)
            return tokens

        if self.bpe_ranks:
            words = re.findall(r"\S+|\s+", text)
            for w in words:
                cached = self._bpe_cache.get(w)
                if cached is not None:
                    tokens.extend(cached)
                elif w in self.vocab:
                    tid = self.vocab[w]
                    self._bpe_cache[w] = [tid]
                    tokens.append(tid)
                else:
                    subwords = self._bpe(w)
                    w_tokens = [self.vocab.get(sw, self.unk_token_id) for sw in subwords]
                    if len(self._bpe_cache) < 100000:
                        self._bpe_cache[w] = w_tokens
                    tokens.extend(w_tokens)
        else:
            words = re.findall(r"\S+|\s+", text)
            for w in words:
                cached = self._bpe_cache.get(w)
                if cached is not None:
                    tokens.extend(cached)
                elif w in self.vocab:
                    tid = self.vocab[w]
                    self._bpe_cache[w] = [tid]
                    tokens.append(tid)
                else:
                    i = 0
                    w_tokens = []
                    while i < len(w):
                        matched = False
                        for length in range(min(32, len(w) - i), 0, -1):
                            sub = w[i : i + length]
                            if sub in self.vocab:
                                w_tokens.append(self.vocab[sub])
                                i += length
                                matched = True
                                break
                        if not matched:
                            w_tokens.append(self.vocab.get(w[i], self.unk_token_id))
                            i += 1
                    if len(self._bpe_cache) < 100000:
                        self._bpe_cache[w] = w_tokens
                    tokens.extend(w_tokens)

        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def encode_stream(
        self,
        text_stream: Any,
        add_special_tokens: bool = False,
    ):
        """
        High-throughput streaming encoder for large document streams and continuous feeds.
        Yields token IDs incrementally without loading entire text chunks into memory.
        """
        if add_special_tokens and self.bos_token_id is not None:
            yield self.bos_token_id

        for chunk in text_stream:
            if not chunk:
                continue
            chunk_tokens = self.encode(chunk, add_special_tokens=False)
            for tid in chunk_tokens:
                yield tid

        if add_special_tokens and self.eos_token_id is not None:
            yield self.eos_token_id

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        if getattr(self, "_native_tok", None) is not None:
            try:
                return self._native_tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
            except Exception:
                pass

        special_ids = {self.bos_token_id, self.eos_token_id, self.unk_token_id, self.pad_token_id}
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            chars.append(self.id_to_token.get(tid, ""))
        return "".join(chars)

    def apply_chat_template(
        self,
        conversation: List[Dict[str, str]],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, List[int], Dict[str, Any]]:
        """
        Formats a multi-turn conversation into a prompt string or token IDs using Jinja2.
        Matches the Hugging Face `tokenizer.apply_chat_template()` standard interface.
        """
        template_str = chat_template
        if template_str is None and "template_name" in kwargs:
            template_str = self.chat_templates.get(kwargs.pop("template_name"))
        if template_str is None:
            template_str = self.chat_template or DEFAULT_CHAT_TEMPLATE

        if HAS_JINJA2:
            try:
                env = jinja2.Environment(
                    trim_blocks=True,
                    lstrip_blocks=True,
                    undefined=jinja2.Undefined,
                )
                template = env.from_string(template_str)
                rendered = template.render(
                    messages=conversation,
                    add_generation_prompt=add_generation_prompt,
                    bos_token=self.bos_token,
                    eos_token=self.eos_token,
                    **kwargs,
                )
            except Exception:
                rendered = _render_chat_template_fallback(
                    conversation,
                    add_generation_prompt=add_generation_prompt,
                    template_str=template_str,
                )
        else:
            rendered = _render_chat_template_fallback(
                conversation,
                add_generation_prompt=add_generation_prompt,
                template_str=template_str,
            )

        if not tokenize:
            return rendered

        token_ids = self.encode(rendered, add_special_tokens=False)
        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(token_ids)), dtype=torch.long),
            }
        return token_ids

    def to_hf_tokenizer(self):
        """Loads and returns the huggingface tokenizers.Tokenizer instance if available."""
        if self.huggingface_json:
            try:
                from tokenizers import Tokenizer
                return Tokenizer.from_str(self.huggingface_json)
            except Exception:
                pass
        return None

    def apply_fim(self, prefix: str, suffix: str, middle: str = "") -> str:
        """Formats a Fill-In-the-Middle (FIM) prompt for code completion."""
        has_special = ("<fim_prefix>" in self.vocab and "<fim_suffix>" in self.vocab and "<fim_middle>" in self.vocab)
        if has_special:
            base = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
            return f"{base}{middle}" if middle else base
        has_qwen = ("<|fim_prefix|>" in self.vocab and "<|fim_suffix|>" in self.vocab and "<|fim_middle|>" in self.vocab)
        if has_qwen:
            base = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
            return f"{base}{middle}" if middle else base
        base = f"<PRE> {prefix} <SUF>{suffix} <MID>"
        return f"{base}{middle}" if middle else base

    def export_to_metadata(self) -> Dict[str, Any]:
        """
        Exports tokenizer parameters into HK metadata format (matching GGUF standard metadata conventions).
        """
        meta = {
            "tokenizer.tokens": json.dumps(self.tokens),
            "tokenizer.scores": json.dumps(self.scores),
            "tokenizer.merges": json.dumps([" ".join(m) for m in self.merges]),
            "tokenizer.chat_template": str(self.chat_template),
            "tokenizer.bos_token": str(self.bos_token),
            "tokenizer.eos_token": str(self.eos_token),
            "tokenizer.unk_token": str(self.unk_token),
            "tokenizer.pad_token": str(self.pad_token),
            "tokenizer.bos_token_id": int(self.bos_token_id),
            "tokenizer.eos_token_id": int(self.eos_token_id),
            "tokenizer.unk_token_id": int(self.unk_token_id),
            "tokenizer.pad_token_id": int(self.pad_token_id),
        }
        if self.chat_templates:
            meta["tokenizer.chat_templates"] = json.dumps(self.chat_templates)
        if self.huggingface_json:
            meta["tokenizer.huggingface.json"] = self.huggingface_json
        return meta

    def __call__(self, text: Union[str, List[str]], return_tensors: Optional[str] = None):
        if isinstance(text, str):
            token_ids = [self.encode(text)]
        else:
            token_ids = [self.encode(t) for t in text]

        max_len = max(len(t) for t in token_ids)
        padded = [t + [self.pad_token_id] * (max_len - len(t)) for t in token_ids]
        mask = [[1] * len(t) + [0] * (max_len - len(t)) for t in token_ids]

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(mask, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": mask}

    def save_to_hk(self, hk_path: Union[str, Path]) -> None:
        """Embeds or patches tokenizer metadata keys directly into an .hk container."""
        from .torch import metadata_set, save_file
        path = Path(hk_path)
        metadata = self.export_to_metadata()
        if path.is_file():
            for k, v in metadata.items():
                metadata_set(str(path), k, v)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Create a lightweight container with tokenizer metadata
            save_file({}, str(path), metadata=metadata)

    def save_pretrained(self, save_directory_or_path: Union[str, Path]) -> str:
        """
        Saves tokenizer to disk.
        If save_directory_or_path ends with .hk, embeds metadata directly in the container.
        Otherwise, writes standard tokenizer.json and patches any .hk files in the directory.
        """
        target = Path(save_directory_or_path)
        if target.suffix.lower() == ".hk":
            self.save_to_hk(target)
            return str(target)

        os.makedirs(target, exist_ok=True)
        vocab_file = target / "tokenizer.json"
        data = {
            "vocab": self.vocab,
            "tokens": self.tokens,
            "scores": self.scores,
            "merges": [" ".join(m) for m in self.merges],
            "chat_template": self.chat_template,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
        }
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Also patch any .hk files found in directory
        hk_files = list(target.glob("*.hk"))
        for hk_file in hk_files:
            try:
                self.save_to_hk(hk_file)
            except Exception:
                pass

        return str(vocab_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs: Any) -> "HKTokenizer":
        path = Path(pretrained_model_name_or_path)

        # 1. Direct .hk binary file inspection (in-file tokenizer metadata!)
        if path.is_file() and path.suffix.lower() == ".hk":
            return cls._from_hk_file(path)

        # 2. Binary SentencePiece .model file
        if path.is_file() and path.suffix.lower() == ".model":
            tokens, scores, token_types = parse_sentencepiece_model(path)
            return cls(tokens=tokens, scores=scores, token_types=token_types)

        # 3. Mistral Tekkenizer tekken.json
        if path.is_file() and path.name == "tekken.json":
            tokens, scores, merges = parse_tekken_json(path)
            return cls(tokens=tokens, scores=scores, merges=merges)

        # 4. Index manifest file (e.g. model.hk.index.json)
        if path.is_file() and path.name.endswith(".index.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                weight_map = idx_data.get("weight_map", {})
                shard_names = sorted(set(weight_map.values()))
                for s_name in shard_names:
                    s_path = path.parent / s_name
                    if s_path.is_file():
                        tok = cls._from_hk_file(s_path)
                        if tok.tokens:
                            return tok
            except Exception:
                pass

        # 5. Directory inspection
        if path.is_dir():
            # Check for index manifest first
            for idx_candidate in [path / "model.hk.index.json", *path.glob("*.index.json")]:
                if idx_candidate.is_file():
                    try:
                        with open(idx_candidate, "r", encoding="utf-8") as f:
                            idx_data = json.load(f)
                        weight_map = idx_data.get("weight_map", {})
                        shard_names = sorted(set(weight_map.values()))
                        for s_name in shard_names:
                            s_path = path / s_name
                            if s_path.is_file():
                                tok = cls._from_hk_file(s_path)
                                if tok.tokens:
                                    return tok
                    except Exception:
                        pass

            # Check for any .hk files
            hk_files = sorted(path.glob("*.hk"))
            for hk_file in hk_files:
                try:
                    tok = cls._from_hk_file(hk_file)
                    if tok.tokens:
                        return tok
                except Exception:
                    pass

            # Check for binary SentencePiece .model
            for spm_candidate in [path / "tokenizer.model", *path.glob("*.model")]:
                if spm_candidate.is_file():
                    try:
                        tokens, scores, token_types = parse_sentencepiece_model(spm_candidate)
                        return cls(tokens=tokens, scores=scores, token_types=token_types)
                    except Exception:
                        pass

            # Check for Mistral Tekkenizer tekken.json
            tekken_file = path / "tekken.json"
            if tekken_file.is_file():
                try:
                    tokens, scores, merges = parse_tekken_json(tekken_file)
                    return cls(tokens=tokens, scores=scores, merges=merges)
                except Exception:
                    pass

            t_file = path / "tokenizer.json"
            if t_file.is_file():
                with open(t_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls(
                    vocab=data.get("vocab"),
                    tokens=data.get("tokens"),
                    scores=data.get("scores"),
                    merges=data.get("merges"),
                    chat_template=data.get("chat_template"),
                    bos_token=data.get("bos_token", "<s>"),
                    eos_token=data.get("eos_token", "</s>"),
                    unk_token=data.get("unk_token", "<unk>"),
                    pad_token=data.get("pad_token", "<pad>"),
                )

        if path.is_file() and path.name == "tokenizer.json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                vocab=data.get("vocab"),
                tokens=data.get("tokens"),
                scores=data.get("scores"),
                merges=data.get("merges"),
                chat_template=data.get("chat_template"),
                bos_token=data.get("bos_token", "<s>"),
                eos_token=data.get("eos_token", "</s>"),
                unk_token=data.get("unk_token", "<unk>"),
                pad_token=data.get("pad_token", "<pad>"),
            )

        return cls()

    @classmethod
    def _from_hk_file(cls, hk_path: Union[str, Path]) -> "HKTokenizer":
        """Reconstructs tokenizer directly from in-file HK metadata without external JSON files."""
        from .torch import HKFile

        with HKFile(hk_path) as f:
            meta = f.metadata()

        tokens = None
        scores = None
        token_types = None
        merges = None
        pre_tokenizer = meta.get("tokenizer.pre")
        chat_template = meta.get("tokenizer.chat_template")

        # Parse tokenizer.tokens
        if "tokenizer.tokens" in meta:
            val = meta["tokenizer.tokens"]
            if isinstance(val, str):
                try:
                    tokens = json.loads(val)
                except Exception:
                    tokens = val.split()
            elif isinstance(val, list):
                tokens = val

        # Parse tokenizer.scores
        if "tokenizer.scores" in meta:
            val = meta["tokenizer.scores"]
            if isinstance(val, str):
                try:
                    scores = json.loads(val)
                except Exception:
                    scores = [float(x) for x in val.split()]
            elif isinstance(val, list):
                scores = [float(x) for x in val]

        # Parse tokenizer.token_types
        if "tokenizer.token_types" in meta:
            val = meta["tokenizer.token_types"]
            if isinstance(val, str):
                try:
                    token_types = json.loads(val)
                except Exception:
                    token_types = [int(x) for x in val.split()]
            elif isinstance(val, list):
                token_types = [int(x) for x in val]

        # Parse tokenizer.merges
        if "tokenizer.merges" in meta:
            val = meta["tokenizer.merges"]
            if isinstance(val, str):
                try:
                    merges = json.loads(val)
                except Exception:
                    merges = [line for line in val.split("\n") if line.strip()]
            elif isinstance(val, list):
                merges = val

        bos_token = meta.get("tokenizer.bos_token", "<s>")
        eos_token = meta.get("tokenizer.eos_token", "</s>")
        unk_token = meta.get("tokenizer.unk_token", "<unk>")
        pad_token = meta.get("tokenizer.pad_token", "<pad>")

        huggingface_json = meta.get("tokenizer.huggingface.json")
        chat_templates = {}
        if "tokenizer.chat_templates" in meta:
            try:
                chat_templates = json.loads(meta["tokenizer.chat_templates"])
            except Exception:
                pass
        if not chat_template and "default" in chat_templates:
            chat_template = chat_templates["default"]

        tok = cls(
            tokens=tokens,
            scores=scores,
            token_types=token_types,
            merges=merges,
            pre_tokenizer=pre_tokenizer,
            chat_template=chat_template,
            chat_templates=chat_templates,
            huggingface_json=huggingface_json,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
        )

        # Reconstruct explicit IDs if present in metadata
        for id_attr, meta_key in [
            ("bos_token_id", "tokenizer.bos_token_id"),
            ("eos_token_id", "tokenizer.eos_token_id"),
            ("unk_token_id", "tokenizer.unk_token_id"),
            ("pad_token_id", "tokenizer.pad_token_id"),
        ]:
            if meta_key in meta:
                try:
                    setattr(tok, id_attr, int(meta[meta_key]))
                except Exception:
                    pass

        # Attach native tokenizer if available
        if is_native_available():
            try:
                tok._native_tok = NativeHKTokenizer(hk_path)
            except Exception:
                tok._native_tok = None

        return tok


class AutoTokenizer:
    """Factory for loading tokenizers automatically from .hk containers, index manifests, or config directories."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs: Any) -> HKTokenizer:
        return HKTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)


