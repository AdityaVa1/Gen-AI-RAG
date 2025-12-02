#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from typing import Dict, List, Iterable, Tuple, Any

START = "<s>"
END = "</s>"
UNK = "<unk>"


# --------------------
# Data reading + OOV mapping
# --------------------

def read_utterances(path: str) -> List[List[str]]:
    utts: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(line.split())
    return utts


def map_oov(tokens: List[str], vocab: set[str]) -> List[str]:
    return [t if t in vocab else UNK for t in tokens]


# --------------------
# Load model JSON
# --------------------

def load_model(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)

    # Extract counts
    uni_counts: Dict[str, int] = data["unigram"]["counts"]

    bi_counts: Dict[Tuple[str, str], int] = {
        tuple(k.split()): v for k, v in data["bigram"]["counts"].items()
    }

    tri_counts: Dict[Tuple[str, str, str], int] = {
        tuple(k.split()): v for k, v in data["trigram"]["counts"].items()
    }

    meta = data["meta"]

    return {
        "meta": meta,
        "unigram": uni_counts,
        "bigram": bi_counts,
        "trigram": tri_counts
    }


# --------------------
# Probability estimators
# --------------------

def p_unigram(w: str, model: Dict[str, Any]) -> float:
    uni = model["unigram"]
    total = model["meta"]["unigram_total"]
    return uni.get(w, 0) / total


def p_bigram(w1: str, w2: str, model: Dict[str, Any], laplace: bool) -> float:
    bi = model["bigram"]
    uni = model["unigram"]
    V = model["meta"]["vocab_size_conditioning"]

    num = bi.get((w1, w2), 0)
    den = uni.get(w1, 0)

    if laplace:
        return (num + 1) / (den + V)

    if den == 0:
        return 0.0
    return num / den


def p_trigram(w1: str, w2: str, w3: str, model: Dict[str, Any], laplace: bool) -> float:
    tri = model["trigram"]
    bi = model["bigram"]
    V = model["meta"]["vocab_size_conditioning"]

    num = tri.get((w1, w2, w3), 0)
    den = bi.get((w1, w2), 0)

    if laplace:
        return (num + 1) / (den + V)

    if den == 0:
        return 0.0
    return num / den


# --------------------
# Perplexity computation
# --------------------

def compute_ppl(
    model_type: str,
    model: Dict[str, Any],
    utterances: List[List[str]],
    laplace: bool,
    vocab: set[str]
) -> float:

    log_prob_sum = 0.0
    N = 0

    for utt in utterances:
        utt = map_oov(utt, vocab)
        seq = [START] + utt + [END]

        if model_type == "unigram":
            for w in seq:
                p = p_unigram(w, model)
                if p == 0:
                    return float("inf")
                log_prob_sum += math.log(p)
                N += 1

        elif model_type == "bigram":
            for i in range(len(seq) - 1):
                p = p_bigram(seq[i], seq[i + 1], model, laplace)
                if p == 0:
                    return float("inf")
                log_prob_sum += math.log(p)
                N += 1

        elif model_type == "trigram":
            for i in range(len(seq) - 2):
                p = p_trigram(seq[i], seq[i + 1], seq[i + 2], model, laplace)
                if p == 0:
                    return float("inf")
                log_prob_sum += math.log(p)
                N += 1

    return math.exp(-log_prob_sum / N)


# --------------------
# main()
# --------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["unigram", "bigram", "trigram"])
    parser.add_argument("model_file")
    parser.add_argument("data_file")
    parser.add_argument("--laplace", action="store_true")
    args = parser.parse_args()

    if args.model_type == "unigram" and args.laplace:
        raise RuntimeError("Unigram model cannot use Laplace smoothing.")

    model = load_model(args.model_file)
    utterances = read_utterances(args.data_file)

    # Build vocab set for OOV mapping
    vocab = set(model["unigram"].keys())

    ppl = compute_ppl(
        model_type=args.model_type,
        model=model,
        utterances=utterances,
        laplace=args.laplace,
        vocab=vocab
    )

    print(ppl)


if __name__ == "__main__":
    main()
