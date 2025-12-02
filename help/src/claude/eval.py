#!/usr/bin/env python3
"""
Evaluate n-gram language models by computing perplexity.

Usage:
    python3 src/eval.py <model_type> <model_file> <data_file> [--laplace]

Arguments:
    model_type: Type of model (unigram, bigram, or trigram)
    model_file: Path to the trained models (JSON format)
    data_file: Path to the data file for evaluation
    --laplace: (optional) Use Laplace (add-one) smoothing
"""

import json
import argparse
import math
import sys


def read_data(filepath):
    """
    Read evaluation data from file.
    Returns a list of utterances, where each utterance is a list of phonemes.
    """
    utterances = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                phonemes = line.split()
                if phonemes:
                    utterances.append(phonemes)
    return utterances


def add_boundary_markers(utterances):
    """
    Add begin-of-utterance (<s>) and end-of-utterance (</s>) markers.
    """
    marked_utterances = []
    for utterance in utterances:
        marked = ['<s>'] + utterance + ['</s>']
        marked_utterances.append(marked)
    return marked_utterances


def compute_unigram_perplexity(model_data, utterances):
    """
    Compute perplexity using unigram model (no smoothing for unigrams).
    """
    unigram_counts = model_data['unigram']
    total_count = sum(unigram_counts.values())
    
    log_prob_sum = 0
    N = 0  # Total number of tokens
    
    for utterance in utterances:
        for phoneme in utterance:
            # Skip boundary markers
            if phoneme in ['<s>', '</s>']:
                continue
            
            N += 1
            count = unigram_counts.get(phoneme, 0)
            
            if count == 0:
                # Handle OOV: assign a very small probability
                # Use 1 / (total_count + 1) as a simple approach
                prob = 1 / (total_count + 1)
            else:
                prob = count / total_count
            
            log_prob_sum += math.log2(prob)
    
    if N == 0:
        return float('inf')
    
    # Perplexity = 2^(-1/N * sum(log2(P(w_i))))
    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity


def compute_bigram_perplexity(model_data, utterances, use_laplace=False):
    """
    Compute perplexity using bigram model.
    """
    bigram_counts = model_data['bigram']['bigrams']
    context_counts = model_data['bigram']['contexts']
    vocab_size = len(model_data['vocabulary'])
    
    log_prob_sum = 0
    N = 0  # Total number of bigrams
    
    for utterance in utterances:
        for i in range(len(utterance) - 1):
            context = utterance[i]
            word = utterance[i + 1]
            N += 1
            
            if use_laplace:
                # Laplace smoothing: P(w|c) = (count(c,w) + 1) / (count(c) + V)
                bigram_count = bigram_counts.get(context, {}).get(word, 0)
                context_count = context_counts.get(context, 0)
                prob = (bigram_count + 1) / (context_count + vocab_size)
            else:
                # No smoothing
                context_count = context_counts.get(context, 0)
                
                if context_count == 0:
                    # Unseen context: use a very small probability
                    prob = 1e-10
                else:
                    bigram_count = bigram_counts.get(context, {}).get(word, 0)
                    if bigram_count == 0:
                        # Unseen bigram: use a very small probability
                        prob = 1e-10
                    else:
                        prob = bigram_count / context_count
            
            log_prob_sum += math.log2(prob)
    
    if N == 0:
        return float('inf')
    
    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity


def compute_trigram_perplexity(model_data, utterances, use_laplace=False):
    """
    Compute perplexity using trigram model.
    """
    trigram_counts = model_data['trigram']['trigrams']
    bigram_context_counts = model_data['trigram']['bigram_contexts']
    vocab_size = len(model_data['vocabulary'])
    
    log_prob_sum = 0
    N = 0  # Total number of trigrams
    
    for utterance in utterances:
        for i in range(len(utterance) - 2):
            w1 = utterance[i]
            w2 = utterance[i + 1]
            w3 = utterance[i + 2]
            N += 1
            
            if use_laplace:
                # Laplace smoothing: P(w3|w1,w2) = (count(w1,w2,w3) + 1) / (count(w1,w2) + V)
                trigram_count = trigram_counts.get(w1, {}).get(w2, {}).get(w3, 0)
                bigram_count = bigram_context_counts.get(w1, {}).get(w2, 0)
                prob = (trigram_count + 1) / (bigram_count + vocab_size)
            else:
                # No smoothing
                bigram_count = bigram_context_counts.get(w1, {}).get(w2, 0)
                
                if bigram_count == 0:
                    # Unseen context: use a very small probability
                    prob = 1e-10
                else:
                    trigram_count = trigram_counts.get(w1, {}).get(w2, {}).get(w3, 0)
                    if trigram_count == 0:
                        # Unseen trigram: use a very small probability
                        prob = 1e-10
                    else:
                        prob = trigram_count / bigram_count
            
            log_prob_sum += math.log2(prob)
    
    if N == 0:
        return float('inf')
    
    perplexity = 2 ** (-log_prob_sum / N)
    return perplexity


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate n-gram language models'
    )
    parser.add_argument('model_type', 
                       choices=['unigram', 'bigram', 'trigram'],
                       help='Type of model to evaluate')
    parser.add_argument('model_file', 
                       help='Path to the trained models (JSON)')
    parser.add_argument('data_file', 
                       help='Path to the evaluation data file')
    parser.add_argument('--laplace', 
                       action='store_true',
                       help='Use Laplace (add-one) smoothing')
    
    args = parser.parse_args()
    
    # Validate: unigram with laplace is invalid
    if args.model_type == 'unigram' and args.laplace:
        raise ValueError("Laplace smoothing is not applicable to unigram models")
    
    # Load models
    with open(args.model_file, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Read evaluation data
    utterances = read_data(args.data_file)
    
    # Add boundary markers
    marked_utterances = add_boundary_markers(utterances)
    
    # Compute perplexity based on model type
    if args.model_type == 'unigram':
        perplexity = compute_unigram_perplexity(model_data, marked_utterances)
    elif args.model_type == 'bigram':
        perplexity = compute_bigram_perplexity(model_data, marked_utterances, args.laplace)
    elif args.model_type == 'trigram':
        perplexity = compute_trigram_perplexity(model_data, marked_utterances, args.laplace)
    
    # Print perplexity
    print(f"{perplexity:.2f}")


if __name__ == '__main__':
    main()
