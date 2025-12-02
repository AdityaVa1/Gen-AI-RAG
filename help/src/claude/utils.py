#!/usr/bin/env python3
"""
Utility functions for n-gram language models.

This module provides helper functions for analyzing models,
computing statistics, and debugging.
"""

import json
import sys
from collections import Counter


def load_model(filepath):
    """Load trained models from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_model_stats(model_data):
    """Print statistics about the trained models."""
    print("Model Statistics")
    print("=" * 60)
    
    # Vocabulary
    vocab_size = len(model_data['vocabulary'])
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of utterances: {model_data['num_utterances']}")
    
    # Unigram
    unigram = model_data['unigram']
    total_tokens = sum(unigram.values())
    print(f"\nUnigram:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Unique types: {len(unigram)}")
    
    # Bigram
    bigram = model_data['bigram']
    bigram_types = sum(len(v) for v in bigram['bigrams'].values())
    bigram_tokens = sum(bigram['contexts'].values())
    print(f"\nBigram:")
    print(f"  Total bigrams: {bigram_tokens}")
    print(f"  Unique types: {bigram_types}")
    
    # Trigram
    trigram = model_data['trigram']
    trigram_types = sum(
        sum(len(v2) for v2 in v1.values()) 
        for v1 in trigram['trigrams'].values()
    )
    trigram_tokens = sum(
        sum(v.values()) 
        for v in trigram['bigram_contexts'].values()
    )
    print(f"\nTrigram:")
    print(f"  Total trigrams: {trigram_tokens}")
    print(f"  Unique types: {trigram_types}")


def print_top_ngrams(model_data, n=10):
    """Print the most frequent n-grams."""
    print("\n" + "=" * 60)
    print(f"Top {n} Most Frequent N-grams")
    print("=" * 60)
    
    # Top unigrams
    unigram = model_data['unigram']
    top_unigrams = sorted(unigram.items(), key=lambda x: x[1], reverse=True)[:n]
    print(f"\nTop {n} Unigrams:")
    for phoneme, count in top_unigrams:
        print(f"  {phoneme}: {count}")
    
    # Top bigrams
    bigram_counts = []
    for context, words in model_data['bigram']['bigrams'].items():
        for word, count in words.items():
            bigram_counts.append(((context, word), count))
    
    top_bigrams = sorted(bigram_counts, key=lambda x: x[1], reverse=True)[:n]
    print(f"\nTop {n} Bigrams:")
    for (w1, w2), count in top_bigrams:
        print(f"  {w1} {w2}: {count}")
    
    # Top trigrams
    trigram_counts = []
    for w1, rest in model_data['trigram']['trigrams'].items():
        for w2, words in rest.items():
            for w3, count in words.items():
                trigram_counts.append(((w1, w2, w3), count))
    
    top_trigrams = sorted(trigram_counts, key=lambda x: x[1], reverse=True)[:n]
    print(f"\nTop {n} Trigrams:")
    for (w1, w2, w3), count in top_trigrams:
        print(f"  {w1} {w2} {w3}: {count}")


def compute_probability(model_data, sequence, model_type='bigram', smoothing=False):
    """
    Compute the probability of a phoneme sequence.
    
    Args:
        model_data: Loaded model data
        sequence: List of phonemes
        model_type: 'unigram', 'bigram', or 'trigram'
        smoothing: Whether to use Laplace smoothing
    
    Returns:
        Log probability (base 2) of the sequence
    """
    import math
    
    # Add boundary markers
    sequence = ['<s>'] + sequence + ['</s>']
    
    log_prob = 0.0
    vocab_size = len(model_data['vocabulary'])
    
    if model_type == 'unigram':
        unigram = model_data['unigram']
        total = sum(unigram.values())
        
        for phoneme in sequence:
            if phoneme in ['<s>', '</s>']:
                continue
            count = unigram.get(phoneme, 0)
            if count == 0:
                prob = 1 / (total + 1)
            else:
                prob = count / total
            log_prob += math.log2(prob)
    
    elif model_type == 'bigram':
        bigram_counts = model_data['bigram']['bigrams']
        context_counts = model_data['bigram']['contexts']
        
        for i in range(len(sequence) - 1):
            context = sequence[i]
            word = sequence[i + 1]
            
            if smoothing:
                bigram_count = bigram_counts.get(context, {}).get(word, 0)
                context_count = context_counts.get(context, 0)
                prob = (bigram_count + 1) / (context_count + vocab_size)
            else:
                context_count = context_counts.get(context, 0)
                if context_count == 0:
                    prob = 1e-10
                else:
                    bigram_count = bigram_counts.get(context, {}).get(word, 0)
                    prob = bigram_count / context_count if bigram_count > 0 else 1e-10
            
            log_prob += math.log2(prob)
    
    elif model_type == 'trigram':
        trigram_counts = model_data['trigram']['trigrams']
        bigram_context_counts = model_data['trigram']['bigram_contexts']
        
        for i in range(len(sequence) - 2):
            w1, w2, w3 = sequence[i], sequence[i + 1], sequence[i + 2]
            
            if smoothing:
                trigram_count = trigram_counts.get(w1, {}).get(w2, {}).get(w3, 0)
                bigram_count = bigram_context_counts.get(w1, {}).get(w2, 0)
                prob = (trigram_count + 1) / (bigram_count + vocab_size)
            else:
                bigram_count = bigram_context_counts.get(w1, {}).get(w2, 0)
                if bigram_count == 0:
                    prob = 1e-10
                else:
                    trigram_count = trigram_counts.get(w1, {}).get(w2, {}).get(w3, 0)
                    prob = trigram_count / bigram_count if trigram_count > 0 else 1e-10
            
            log_prob += math.log2(prob)
    
    return log_prob


def main():
    """Command-line interface for utility functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='N-gram model utilities')
    parser.add_argument('model_file', help='Path to trained model (JSON)')
    parser.add_argument('--stats', action='store_true', 
                       help='Print model statistics')
    parser.add_argument('--top', type=int, default=10, 
                       help='Show top N n-grams (default: 10)')
    parser.add_argument('--prob', nargs='+', 
                       help='Compute probability of a sequence')
    parser.add_argument('--model-type', choices=['unigram', 'bigram', 'trigram'],
                       default='bigram', help='Model type for probability')
    parser.add_argument('--laplace', action='store_true',
                       help='Use Laplace smoothing for probability')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_file}...")
    model_data = load_model(args.model_file)
    
    if args.stats:
        print_model_stats(model_data)
        print_top_ngrams(model_data, args.top)
    
    if args.prob:
        sequence = args.prob
        log_prob = compute_probability(
            model_data, sequence, args.model_type, args.laplace
        )
        prob = 2 ** log_prob
        print(f"\nSequence: {' '.join(sequence)}")
        print(f"Model: {args.model_type}" + 
              (" (Laplace)" if args.laplace else " (unsmoothed)"))
        print(f"Log probability (base 2): {log_prob:.4f}")
        print(f"Probability: {prob:.2e}")


if __name__ == '__main__':
    main()
