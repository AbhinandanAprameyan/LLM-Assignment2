import argparse
import hashlib
import html
import json
import logging
import os
import pickle
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

# ---------------------------------------------------------------------------
# Optional heavy dependencies. We degrade gracefully if they are missing so
# the pipeline logic can still be exercised in constrained / offline
# environments (see module docstring above).
# ---------------------------------------------------------------------------
try:
    from datasets import load_dataset  # Hugging Face `datasets`
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    HF_TOKENIZERS_AVAILABLE = True
except ImportError:
    HF_TOKENIZERS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset, IterableDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:  # tiny no-op fallback so the script never hard-fails
    def tqdm(iterable, **kwargs):
        return iterable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")


# ===========================================================================
# 1. DATA COLLECTION
# ===========================================================================

@dataclass
class RawDocument:
    """A single raw document plus lightweight provenance metadata."""
    text: str
    domain: str
    source: str


# ---- 1a. Real collection path (Hugging Face `datasets`, streaming) -------

# Each domain lists one or more (HF dataset name, HF config/subset, split,
# text field) candidates, tried in order. Hugging Face reorganized several
# community datasets under canonical namespaces (e.g. "wikitext" ->
# "Salesforce/wikitext") in 2024; newer versions of `datasets`/
# `huggingface_hub` require the namespaced id and will raise `HfUriError`
# for the old bare name. Listing multiple candidates per domain lets the
# script keep working regardless of which naming convention the installed
# library version expects, without the user needing to look anything up.
REAL_DATASET_SOURCES = {
    "encyclopedic": [
        ("Salesforce/wikitext", "wikitext-103-raw-v1", "train", "text"),
        ("wikitext", "wikitext-103-raw-v1", "train", "text"),
        ("wikimedia/wikipedia", "20231101.en", "train", "text"),
    ],
    "news": [
        ("fancyzhx/ag_news", None, "train", "text"),
        ("ag_news", None, "train", "text"),
        ("cc_news", None, "train", "text"),
    ],
    "web": [
        ("Skylion007/openwebtext", None, "train", "text"),
        ("openwebtext", None, "train", "text"),
        ("stas/openwebtext-10k", None, "train", "text"),
    ],
}


def _stream_domain(domain: str, candidates, target_bytes: int) -> List[RawDocument]:
    """
    Try each (name, config, split, text_field) candidate for a domain, in
    order, until one loads successfully via `datasets.load_dataset(...,
    streaming=True)`. Returns as soon as `target_bytes` of raw text has
    been collected for this domain.
    """
    last_error: Optional[Exception] = None
    for name, config, split, text_field in candidates:
        try:
            log.info("Streaming %s (%s) for domain '%s' ...", name, config, domain)
            ds = load_dataset(name, config, split=split, streaming=True)
        except Exception as exc:  # noqa: BLE001 - deliberately broad: try next candidate
            log.warning("  -> could not load %s (%s): %s", name, config, exc)
            last_error = exc
            continue

        docs: List[RawDocument] = []
        domain_bytes = 0
        try:
            for example in ds:
                text = example.get(text_field, "")
                if not text:
                    continue
                docs.append(RawDocument(text=text, domain=domain, source=name))
                domain_bytes += len(text.encode("utf-8"))
                if domain_bytes >= target_bytes:
                    break
        except Exception as exc:  # noqa: BLE001 - streaming failed partway through
            log.warning("  -> streaming from %s failed partway through: %s", name, exc)
            last_error = exc
            if docs:  # keep whatever we already collected rather than discarding it
                log.info("  -> keeping %.2f MB collected before the failure", domain_bytes / 1e6)
                return docs
            continue

        log.info("  -> collected %.2f MB for domain '%s' from %s", domain_bytes / 1e6, domain, name)
        return docs

    raise RuntimeError(
        f"All dataset candidates failed for domain '{domain}'. Last error: {last_error}"
    )


def collect_real_datasets(target_bytes: int) -> List[RawDocument]:
    """
    Stream documents from several public Hugging Face datasets (Wikipedia
    text, news, and general web text) until `target_bytes` of raw text has
    been collected, roughly balancing the three domains.

    Uses `streaming=True` so that even very large datasets (e.g.
    OpenWebText, tens of GB) never need to be downloaded/materialized in
    full -- we just pull documents until we have enough.
    """
    if not HF_DATASETS_AVAILABLE:
        raise RuntimeError(
            "The `datasets` library is required for the real collection "
            "path. Install it with `pip install datasets` and re-run "
            "without --demo."
        )

    docs: List[RawDocument] = []
    bytes_per_domain_target = target_bytes // len(REAL_DATASET_SOURCES)

    for domain, candidates in REAL_DATASET_SOURCES.items():
        domain_docs = _stream_domain(domain, candidates, bytes_per_domain_target)
        docs.extend(domain_docs)

    total_mb = sum(len(d.text.encode("utf-8")) for d in docs) / 1e6
    log.info("Total raw text collected: %.2f MB across %d documents", total_mb, len(docs))
    return docs


# ---- 1b. Offline fallback corpus (self-authored, synthetic) --------------
#
# The paragraphs below are original text written for this assignment (not
# copied from any external source). They exist ONLY so the cleaning /
# tokenization / DataLoader logic can be exercised end-to-end without
# internet access. Domains mirror the three real sources above
# (encyclopedic, news, web) so downstream code does not need to branch.

_DEMO_ENCYCLOPEDIC = [
    "Photosynthesis is the process by which green plants, algae, and some "
    "bacteria convert light energy into chemical energy stored in glucose. "
    "The process takes place mainly in the chloroplasts of plant cells and "
    "requires carbon dioxide, water, and sunlight as inputs.",
    "The Amazon rainforest spans roughly six million square kilometers "
    "across South America and is home to an estimated ten percent of all "
    "known species on Earth. It plays a significant role in regulating the "
    "global carbon cycle.",
    "A transformer is a type of neural network architecture introduced in "
    "2017 that relies on a mechanism called self-attention to weigh the "
    "relevance of different parts of an input sequence when producing an "
    "output.",
    "Mount Kilimanjaro is a dormant volcano in Tanzania and the highest "
    "mountain in Africa, rising about five thousand eight hundred and "
    "ninety five meters above sea level.",
    "The printing press, developed in the fifteenth century, dramatically "
    "reduced the cost of producing books and is widely credited with "
    "accelerating the spread of literacy across Europe.",
]

_DEMO_NEWS = [
    "City council members voted on Tuesday to approve funding for a new "
    "public transit line connecting the northern suburbs to the downtown "
    "business district, with construction expected to begin next spring.",
    "Local farmers reported a stronger than expected harvest this year, "
    "citing favorable rainfall in the early growing season as the primary "
    "factor behind the improved yields.",
    "The regional science fair drew a record number of student entries "
    "this year, with projects ranging from renewable energy demonstrations "
    "to machine learning experiments built on inexpensive hardware.",
    "Officials announced an expansion of the community library's hours "
    "following a petition signed by more than two thousand residents "
    "requesting later weekday access.",
    "A newly opened community garden has already produced its first batch "
    "of vegetables, which organizers plan to distribute to a local food "
    "bank later this month.",
]

_DEMO_WEB = [
    "Just got back from a weekend hiking trip and wanted to share some "
    "notes for anyone planning a similar route -- bring more water than "
    "you think you need, the last stretch has almost no shade.",
    "Does anyone have recommendations for a lightweight laptop stand for "
    "travel? Looking for something that folds flat and doesn't add much "
    "weight to my bag.",
    "Finally finished refactoring my side project's data pipeline this "
    "weekend. Splitting the cleaning step from the tokenization step made "
    "debugging so much easier.",
    "Tried a new sourdough recipe today and the crumb came out way more "
    "open than my usual attempts. Sharing the hydration ratio in case it "
    "helps anyone else experimenting with their starter.",
    "Quick PSA for anyone setting up a home network closet: label your "
    "cables before you tuck them away, future you will be grateful.",
]


def collect_demo_corpus(repeat: int = 4, inject_duplicates: bool = True) -> List[RawDocument]:
    """
    Build a small, self-authored, multi-domain corpus for offline testing.

    `repeat` controls how many times each base paragraph is duplicated
    (with light variation) to give the pipeline enough volume to
    meaningfully exercise batching. `inject_duplicates` additionally
    inserts a handful of EXACT duplicate documents so the deduplication
    step has something concrete to remove -- this mirrors the kind of
    exact-duplicate noise commonly found in raw web-scraped corpora.
    """
    domains = {
        "encyclopedic": _DEMO_ENCYCLOPEDIC,
        "news": _DEMO_NEWS,
        "web": _DEMO_WEB,
    }
    docs: List[RawDocument] = []
    for domain, paragraphs in domains.items():
        for i in range(repeat):
            for p in paragraphs:
                # Light variation per repeat so not every doc is identical,
                # while still being clearly synthetic / short (fast to run).
                variant = p if i == 0 else f"{p} (session {i})"
                docs.append(RawDocument(text=variant, domain=domain, source="demo"))

    if inject_duplicates:
        # Deliberately duplicate a few documents verbatim so the
        # deduplication step below has real work to do.
        for domain, paragraphs in domains.items():
            docs.append(RawDocument(text=paragraphs[0], domain=domain, source="demo"))
            docs.append(RawDocument(text=paragraphs[0], domain=domain, source="demo"))

        # Also add a couple of very short / low-quality documents so the
        # length filter below has something to remove.
        docs.append(RawDocument(text="ok thanks", domain="web", source="demo"))
        docs.append(RawDocument(text="<p>n/a</p>", domain="web", source="demo"))

    random.shuffle(docs)
    total_mb = sum(len(d.text.encode("utf-8")) for d in docs) / 1e6
    log.info(
        "[DEMO MODE] Built synthetic corpus: %d documents, %.4f MB "
        "(NOT a substitute for the real >=1GB run)",
        len(docs), total_mb,
    )
    return docs


# ===========================================================================
# 2. CLEANING & NORMALIZATION
# ===========================================================================

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")   # [text](url) -> text
_MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`#>]+")            # *, _, `, #, >
_REFERENCE_MARKER_RE = re.compile(r"\[\d+\]|\[citation needed\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_document(text: str, lowercase: bool = False) -> str:
    """
    Normalize a single raw document:
      - unescape HTML entities (e.g. `&amp;` -> `&`)
      - strip HTML tags
      - collapse markdown links to their visible text and drop stray
        markdown emphasis / heading characters
      - remove Wikipedia-style reference markers such as `[12]`
      - collapse repeated whitespace to single spaces and strip ends
      - optionally lowercase (see report for the case-sensitivity
        trade-off discussion)
    """
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MARKDOWN_LINK_RE.sub(r"\1", text)
    text = _MARKDOWN_EMPHASIS_RE.sub(" ", text)
    text = _REFERENCE_MARKER_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def hash_text(text: str) -> str:
    """Stable hash used to detect exact-duplicate documents after cleaning."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_and_deduplicate(
    docs: List[RawDocument],
    min_words: int = 50,
    lowercase: bool = False,
) -> List[RawDocument]:
    """
    Apply cleaning to every document, drop exact duplicates (by hash of the
    cleaned text), and drop documents shorter than `min_words`.

    Returns a new list; also logs how many documents were removed at each
    stage so the report can cite concrete numbers.
    """
    seen_hashes = set()
    cleaned: List[RawDocument] = []
    n_dupes = 0
    n_too_short = 0

    for doc in docs:
        text = clean_document(doc.text, lowercase=lowercase)
        h = hash_text(text)
        if h in seen_hashes:
            n_dupes += 1
            continue
        if len(text.split()) < min_words:
            n_too_short += 1
            continue
        seen_hashes.add(h)
        cleaned.append(RawDocument(text=text, domain=doc.domain, source=doc.source))

    log.info(
        "Cleaning summary: %d input docs -> %d kept | %d exact duplicates "
        "removed | %d too-short (< %d words) removed",
        len(docs), len(cleaned), n_dupes, n_too_short, min_words,
    )
    return cleaned


# ===========================================================================
# 3. TOKENIZATION
# ===========================================================================

class FallbackTokenizer:
    """
    A minimal, dependency-free, whitespace + punctuation level tokenizer
    used ONLY when `transformers` is not installed. It builds its own small
    vocabulary from the corpus so the rest of the pipeline (chunking,
    padding, batching) can be exercised without internet access. This is
    NOT a byte-pair-encoding tokenizer and is not suitable for real model
    pretraining -- the real run uses `transformers.AutoTokenizer` (GPT-2
    byte-level BPE) instead.
    """

    def __init__(self):
        self.token_to_id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.id_to_token: Dict[int, str] = {0: "<pad>", 1: "<unk>"}
        self.pad_token_id = 0

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def build_vocab(self, texts: Iterable[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(self._tokens(text))
        for token, _ in counter.most_common():
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        return [self.token_to_id.get(tok, 1) for tok in self._tokens(text)]

    def __len__(self) -> int:
        return len(self.token_to_id)


def build_tokenizer(model_name: str = "gpt2"):
    """
    Returns a tokenizer object exposing `.encode(text) -> List[int]` and
    `len(tokenizer)` for vocab size, using the real Hugging Face tokenizer
    when available, or the dependency-free fallback otherwise.
    """
    if HF_TOKENIZERS_AVAILABLE:
        log.info("Loading Hugging Face tokenizer: %s", model_name)
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token  # GPT-2 has no pad token by default
        return tok
    else:
        log.warning(
            "`transformers` not available -- using dependency-free "
            "FallbackTokenizer (word-level, NOT BPE). Install "
            "`transformers` for the real run."
        )
        return None  # built by caller once the corpus is known (needs vocab)


@dataclass
class TokenizedExample:
    input_ids: List[int]
    domain: str


def tokenize_and_chunk(
    docs: List[RawDocument],
    tokenizer,
    block_size: int = 512,
    stride: int = 0,
) -> List[TokenizedExample]:
    """
    Tokenize every cleaned document and split it into fixed-length blocks
    of `block_size` tokens (the model's maximum context length). Documents
    longer than `block_size` are chunked into multiple blocks; `stride` > 0
    creates overlapping blocks (a common trick to avoid losing information
    at chunk boundaries). The final partial block of each document is kept
    and padded later by the DataLoader's collate function, rather than
    silently dropped, so no data is discarded.
    """
    examples: List[TokenizedExample] = []
    step = block_size - stride if stride < block_size else block_size

    for doc in docs:
        if hasattr(tokenizer, "encode") and not isinstance(tokenizer, FallbackTokenizer):
            ids = tokenizer.encode(doc.text)
        else:
            ids = tokenizer.encode(doc.text)

        if len(ids) == 0:
            continue

        for start in range(0, len(ids), step):
            chunk = ids[start:start + block_size]
            if len(chunk) == 0:
                continue
            examples.append(TokenizedExample(input_ids=chunk, domain=doc.domain))
            if start + block_size >= len(ids):
                break

    log.info(
        "Tokenization summary: %d cleaned documents -> %d token blocks "
        "(block_size=%d, stride=%d)",
        len(docs), len(examples), block_size, stride,
    )
    return examples


# ===========================================================================
# 4. CUSTOM DATA LOADER
# ===========================================================================

if TORCH_AVAILABLE:

    class TokenizedTextDataset(Dataset):
        """
        Map-style PyTorch Dataset over a list of already-tokenized blocks.
        Suitable when the tokenized corpus comfortably fits in memory.
        """

        def __init__(self, examples: List[TokenizedExample]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> Dict[str, List[int]]:
            ex = self.examples[idx]
            return {"input_ids": ex.input_ids, "domain": ex.domain}

    class StreamingTokenizedDataset(IterableDataset):
        """
        IterableDataset variant that tokenizes documents lazily, one at a
        time, from a generator of cleaned documents. This avoids ever
        materializing the full tokenized corpus in memory, which is the
        approach used for corpora too large to fit in RAM (the assignment's
        "memory bottleneck" requirement).
        """

        def __init__(self, doc_iterable: Iterable[RawDocument], tokenizer, block_size: int):
            self.doc_iterable = doc_iterable
            self.tokenizer = tokenizer
            self.block_size = block_size

        def __iter__(self) -> Iterator[Dict[str, List[int]]]:
            for doc in self.doc_iterable:
                ids = self.tokenizer.encode(doc.text)
                for start in range(0, len(ids), self.block_size):
                    chunk = ids[start:start + self.block_size]
                    if chunk:
                        yield {"input_ids": chunk, "domain": doc.domain}

    def make_collate_fn(pad_token_id: int):
        """
        Pads a batch of variable-length `input_ids` (and builds a matching
        attention mask) to the length of the longest sequence in the batch.
        Left un-padded sequences are truncated to nothing shorter than the
        model's block size, so no example is ever padded beyond necessity.
        """

        def collate_fn(batch: List[Dict]) -> Dict[str, "torch.Tensor"]:
            max_len = max(len(ex["input_ids"]) for ex in batch)
            input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            domains = []
            for i, ex in enumerate(batch):
                length = len(ex["input_ids"])
                input_ids[i, :length] = torch.tensor(ex["input_ids"], dtype=torch.long)
                attention_mask[i, :length] = 1
                domains.append(ex["domain"])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "domain": domains,
            }

        return collate_fn

    def build_dataloader(
        examples: List[TokenizedExample],
        pad_token_id: int,
        batch_size: int = 8,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TokenizedTextDataset(examples)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=make_collate_fn(pad_token_id),
        )

else:
    # ------------------------------------------------------------------
    # Dependency-free fallback Dataset / DataLoader, used ONLY when torch
    # is not installed. Mirrors the torch API surface used above
    # (`__len__`, `__getitem__`, batch iteration with shuffling and
    # padding) with plain Python lists standing in for tensors.
    # ------------------------------------------------------------------

    class TokenizedTextDataset:
        def __init__(self, examples: List[TokenizedExample]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> Dict:
            ex = self.examples[idx]
            return {"input_ids": ex.input_ids, "domain": ex.domain}

    class FallbackDataLoader:
        """Pure-Python batching + shuffling + padding, no torch tensors."""

        def __init__(self, dataset, pad_token_id: int, batch_size: int = 8, shuffle: bool = True):
            self.dataset = dataset
            self.pad_token_id = pad_token_id
            self.batch_size = batch_size
            self.shuffle = shuffle

        def __iter__(self):
            indices = list(range(len(self.dataset)))
            if self.shuffle:
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch = [self.dataset[i] for i in batch_idx]
                max_len = max(len(ex["input_ids"]) for ex in batch)
                input_ids, attention_mask, domains = [], [], []
                for ex in batch:
                    ids = ex["input_ids"]
                    pad_len = max_len - len(ids)
                    input_ids.append(ids + [self.pad_token_id] * pad_len)
                    attention_mask.append([1] * len(ids) + [0] * pad_len)
                    domains.append(ex["domain"])
                yield {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "domain": domains,
                }

    def build_dataloader(
        examples: List[TokenizedExample],
        pad_token_id: int,
        batch_size: int = 8,
        shuffle: bool = True,
    ):
        dataset = TokenizedTextDataset(examples)
        return FallbackDataLoader(dataset, pad_token_id, batch_size, shuffle)


# ===========================================================================
# 5. SAMPLE OUTPUT
# ===========================================================================

def save_sample_batches(dataloader, out_path: str, n_batches: int = 8) -> None:
    """
    Materialize the first `n_batches` batches from `dataloader` and save
    them to `out_path`. When torch is available, saves genuine torch
    tensors via `torch.save` (loadable with `torch.load(out_path)`).
    Otherwise, falls back to `pickle` with plain Python lists in the same
    dict structure so the file can still be loaded and inspected.
    """
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        batches.append(batch)

    if TORCH_AVAILABLE:
        torch.save(batches, out_path)
        log.info("Saved %d real torch batches to %s", len(batches), out_path)
    else:
        with open(out_path, "wb") as f:
            pickle.dump(batches, f)
        log.warning(
            "torch not available -- saved %d batches to %s using plain "
            "pickle (NOT a real .pt torch file). Re-run with torch "
            "installed to produce a genuine torch.save() file.",
            len(batches), out_path,
        )


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def run_pipeline(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    # ---- 1. Collect --------------------------------------------------
    if args.demo:
        raw_docs = collect_demo_corpus()
    else:
        target_bytes = int(args.target_gb * (1024 ** 3))
        raw_docs = collect_real_datasets(target_bytes)

    # ---- 2. Clean & deduplicate ---------------------------------------
    # NOTE: the synthetic demo paragraphs are intentionally short (they only
    # need to exercise the pipeline, not simulate a real corpus), so in
    # --demo mode we relax the length filter unless the user explicitly
    # overrides --min-words. The real run keeps the assignment's default
    # of 50 words.
    effective_min_words = args.min_words
    if args.demo and args.min_words == 50:
        effective_min_words = 15
    clean_docs = clean_and_deduplicate(
        raw_docs, min_words=effective_min_words, lowercase=args.lowercase
    )

    # ---- 3. Tokenize & chunk -------------------------------------------
    tokenizer = build_tokenizer(args.tokenizer)
    if tokenizer is None:  # fallback path: build a vocab from this corpus
        tokenizer = FallbackTokenizer()
        tokenizer.build_vocab(d.text for d in clean_docs)

    examples = tokenize_and_chunk(
        clean_docs, tokenizer, block_size=args.block_size, stride=args.stride
    )

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0

    # ---- 4. Build DataLoader & save sample batches ---------------------
    dataloader = build_dataloader(
        examples, pad_token_id=pad_id, batch_size=args.batch_size, shuffle=True
    )
    sample_path = os.path.join(args.out_dir, "sample_dataset.pt")
    save_sample_batches(dataloader, sample_path, n_batches=args.n_sample_batches)

    # ---- Stats dump (used to populate the accompanying report) ---------
    vocab_size = len(tokenizer) if hasattr(tokenizer, "__len__") else None
    stats = {
        "mode": "demo" if args.demo else "full",
        "raw_documents": len(raw_docs),
        "raw_mb": round(sum(len(d.text.encode("utf-8")) for d in raw_docs) / 1e6, 4),
        "clean_documents": len(clean_docs),
        "clean_mb": round(sum(len(d.text.encode("utf-8")) for d in clean_docs) / 1e6, 4),
        "token_blocks": len(examples),
        "block_size": args.block_size,
        "vocab_size": vocab_size,
        "tokenizer_backend": "huggingface" if HF_TOKENIZERS_AVAILABLE else "fallback_wordlevel",
        "torch_backend": "torch" if TORCH_AVAILABLE else "fallback_pure_python",
        "domains": dict(Counter(d.domain for d in clean_docs)),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    stats_path = os.path.join(args.out_dir, "run_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log.info("Pipeline complete in %.2fs. Stats written to %s", stats["elapsed_seconds"], stats_path)
    log.info(json.dumps(stats, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true",
                   help="Run the offline smoke-test on a small synthetic corpus instead of the real >=1GB collection.")
    p.add_argument("--target-gb", type=float, default=1.0,
                   help="Target raw text size in GB for the real collection run (default: 1.0).")
    p.add_argument("--tokenizer", type=str, default="gpt2",
                   help="Hugging Face tokenizer name to use when `transformers` is available (default: gpt2).")
    p.add_argument("--block-size", type=int, default=512,
                   help="Maximum sequence length (tokens) per training example (default: 512).")
    p.add_argument("--stride", type=int, default=0,
                   help="Overlap (in tokens) between consecutive chunks of a long document (default: 0, no overlap).")
    p.add_argument("--min-words", type=int, default=50,
                   help="Minimum word count for a document to be kept (default: 50).")
    p.add_argument("--lowercase", action="store_true",
                   help="Lowercase all text during cleaning (off by default -- see report for trade-off discussion).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="DataLoader batch size (default: 8).")
    p.add_argument("--n-sample-batches", type=int, default=8,
                   help="Number of batches to materialize into sample_dataset.pt (default: 8).")
    p.add_argument("--out-dir", type=str, default="./output",
                   help="Directory to write sample_dataset.pt and run_stats.json into.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
