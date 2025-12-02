#!/usr/bin/env python3
"""
Train n-gram language models (unigram, bigram, trigram) from phonetic sequence data.

Usage:
    python3 src/train.py <input_file> <output_file>

Arguments:
    input_file: Path to the training data (one utterance per line)
    output_file: Path to save the trained models (JSON format)
"""

import json
import argparse
from collections import defaultdict, Counter
import sys


def read_data(filepath):
    """
    Read training data from file.
    Returns a list of utterances, where each utterance is a list of phonemes.
    """
    utterances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split the line into phonemes
                phonemes = line.split()
                if phonemes:
                    utterances.append(phonemes)
    return utterances


def add_boundary_markers(utterances):
    """
    Add begin-of-utterance (<s>) and end-of-utterance (</s>) markers.
    Returns modified utterances with boundary markers.
    """
    marked_utterances = []
    for utterance in utterances:
        # Add <s> at the beginning and </s> at the end
        marked = ['<s>'] + utterance + ['</s>']
        marked_utterances.append(marked)
    return marked_utterances


def train_unigram(utterances):
    """
    Train a unigram model.
    Returns a dictionary of unigram counts.
    """
    unigram_counts = Counter()
    
    for utterance in utterances:
        for phoneme in utterance:
            # Skip boundary markers for unigram
            if phoneme not in ['<s>', '</s>']:
                unigram_counts[phoneme] += 1
    
    return dict(unigram_counts)


def train_bigram(utterances):
    """
    Train a bigram model.
    Returns a dictionary of bigram counts and context counts.
    """
    bigram_counts = defaultdict(Counter)
    context_counts = Counter()
    
    for utterance in utterances:
        for i in range(len(utterance) - 1):
            context = utterance[i]
            word = utterance[i + 1]
            bigram_counts[context][word] += 1
            context_counts[context] += 1
    
    # Convert defaultdict to regular dict for JSON serialization
    result = {
        'bigrams': {k: dict(v) for k, v in bigram_counts.items()},
        'contexts': dict(context_counts)
    }
    
    return result


def train_trigram(utterances):
    """
    Train a trigram model.
    Returns a dictionary of trigram counts and bigram context counts.
    """
    trigram_counts = defaultdict(lambda: defaultdict(Counter))
    bigram_context_counts = defaultdict(Counter)
    
    for utterance in utterances:
        for i in range(len(utterance) - 2):
            w1 = utterance[i]
            w2 = utterance[i + 1]
            w3 = utterance[i + 2]
            
            trigram_counts[w1][w2][w3] += 1
            bigram_context_counts[w1][w2] += 1
    
    # Convert nested defaultdicts to regular dicts for JSON serialization
    result = {
        'trigrams': {
            k1: {k2: dict(v2) for k2, v2 in v1.items()}
            for k1, v1 in trigram_counts.items()
        },
        'bigram_contexts': {
            k: dict(v) for k, v in bigram_context_counts.items()
        }
    }
    
    return result


def get_vocabulary(utterances):
    """
    Get the vocabulary (set of unique phonemes) from the utterances.
    """
    vocab = set()
    for utterance in utterances:
        for phoneme in utterance:
            if phoneme not in ['<s>', '</s>']:
                vocab.add(phoneme)
    return sorted(list(vocab))


def main():
    parser = argparse.ArgumentParser(
        description='Train n-gram language models from phonetic sequences'
    )
    parser.add_argument('input_file', help='Path to the training data file')
    parser.add_argument('output_file', help='Path to save the trained models (JSON)')
    
    args = parser.parse_args()
    
    print(f"Reading training data from {args.input_file}...")
    utterances = read_data(args.input_file)
    print(f"Loaded {len(utterances)} utterances")
    
    # Add boundary markers
    print("Adding boundary markers...")
    marked_utterances = add_boundary_markers(utterances)
    
    # Get vocabulary
    vocab = get_vocabulary(marked_utterances)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Train models
    print("Training unigram model...")
    unigram_model = train_unigram(marked_utterances)
    
    print("Training bigram model...")
    bigram_model = train_bigram(marked_utterances)
    
    print("Training trigram model...")
    trigram_model = train_trigram(marked_utterances)
    
    # Prepare output
    models = {
        'unigram': unigram_model,
        'bigram': bigram_model,
        'trigram': trigram_model,
        'vocabulary': vocab,
        'num_utterances': len(utterances)
    }
    
    # Save to JSON
    print(f"Saving models to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(models, f, indent=2)
    
    print("Training complete!")
    print(f"Unigram types: {len(unigram_model)}")
    print(f"Bigram types: {sum(len(v) for v in bigram_model['bigrams'].values())}")
    print(f"Trigram types: {sum(sum(len(v2) for v2 in v1.values()) for v1 in trigram_model['trigrams'].values())}")


if __name__ == '__main__':
    main()
