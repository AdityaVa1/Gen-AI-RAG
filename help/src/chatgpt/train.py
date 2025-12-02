#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List, Iterable, Tuple, Any

START = "<s>"
END = "</s>"
UNK = "<unk>"


def read_utterances(path: str) -> List[List[str]]:
    """Reads a data file. Each line is one space-separated utterance."""
    utts: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(line.split())
    return utts


# --------------------
# OOV handling
# --------------------

def build_vocab(utterances: List[List[str]]) -> List[str]:
    """Singletons → <unk> as per J&M Chapter 3."""
    counts = Counter()

    for utt in utterances:
        counts.update(utt)

    vocab = [tok for tok, c in counts.items() if c > 1]
    vocab.append(UNK)
    return vocab


def replace_singletons_with_unk(
    utterances: List[List[str]],
    vocab: List[str]
) -> List[List[str]]:
    vocab_set = set(vocab)
    return [
        [tok if tok in vocab_set else UNK for tok in utt]
        for utt in utterances
    ]


# --------------------
# N-gram counting
# --------------------

def count_ngrams(
    utterances: List[List[str]]
) -> Tuple[
    Counter[str],
    Counter[Tuple[str, str]],
    Counter[Tuple[str, str, str]]
]:
    uni: Counter[str] = Counter()
    bi: Counter[Tuple[str, str]] = Counter()
    tri: Counter[Tuple[str, str, str]] = Counter()

    for utt in utterances:
        seq = [START] + utt + [END]

        uni.update(seq)

        for i in range(len(seq) - 1):
            bi[(seq[i], seq[i + 1])] += 1

        for i in range(len(seq) - 2):
            tri[(seq[i], seq[i + 1], seq[i + 2])] += 1

    return uni, bi, tri


# --------------------
# JSON formatting
# --------------------

def convert_to_json(counter: Counter) -> Dict[str, int]:
    """Convert tuple keys → 'w1 w2' strings."""
    out: Dict[str, int] = {}
    for k, v in counter.items():
        if isinstance(k, tuple):
            out[" ".join(k)] = v
        else:
            out[k] = v
    return out


def create_model_json(
    uni: Counter[str],
    bi: Counter[Tuple[str, str]],
    tri: Counter[Tuple[str, str, str]],
    vocab: List[str],
    n_utterances: int
) -> Dict[str, Any]:

    V = len(vocab)

    return {
        "meta": {
            "description": "n-gram counts without smoothing",
            "bos": START,
            "eos": END,
            "unk": UNK,
            "n_sentences": n_utterances,
            "unigram_total": sum(uni.values()),
            "vocab_size_conditioning": V
        },
        "unigram": {
            "order": 1,
            "counts": convert_to_json(uni)
        },
        "bigram": {
            "order": 2,
            "counts": convert_to_json(bi)
        },
        "trigram": {
            "order": 3,
            "counts": convert_to_json(tri)
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()

    # Read raw data
    raw_utts = read_utterances(args.input_file)

    # OOV handling
    vocab = build_vocab(raw_utts)
    processed_utts = replace_singletons_with_unk(raw_utts, vocab)

    # Count n-grams
    uni, bi, tri = count_ngrams(processed_utts)

    # Build JSON model structure
    model = create_model_json(
        uni=uni,
        bi=bi,
        tri=tri,
        vocab=vocab,
        n_utterances=len(processed_utts)
    )

    # Save
    with open(args.output_file, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model saved to {args.output_file}")


if __name__ == "__main__":
    main()
