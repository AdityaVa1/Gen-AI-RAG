# Answer these questions with respect to the data in the `training` set.

## Does the LLM handle begin and end of sentence tokens across models correctly?

```
The way ChatGPT handles the start and end of utterances is incorrect as it uses the following code:
```python
    for utt in utterances:
        seq = [START] + utt + [END]

        uni.update(seq)
```
Which doesn't account for the N number of starting tokens for a specific N gram. For example, a tri gram should have 3 start tokens and 3 end tokens. This leads to incorrect probabilities. In an N-gram model, the probability of a word is conditional on the previous N−1 words. Therefore, to predict the very first word of a sentence, you need N−1 start tokens so the math works out. 

Take the example of an example from the json files:
Ours: "<s> <s> DH" 
LLM: "<s> DH"
The model cannot compute P(DH∣<s>,?) and hence, loses the ability to model the beginning of the sentence correctly.
```

### Does your code handle these tokens in the same way?

```
No our code doesn't work in the same way. Our code adds N-1 start tokens to the beginning of sentences, but only 1 end token (as once we predict it, we need not generate any other probabilities). This is much more correct in terms of probabilities for prediction. 
```

----


## Does the LLM count n-grams correctly?

```
Yes it does, you can tell by looking at the uni, bi, and tri gram counts in the json files for the tiny_training.txt files. <s> HH as a count 3, which is the number of HH phonemes at the beginning of the utterances in the file. 
```

### Does your code count n-grams in the same way?

```
For the unigram counts, we have the same counts as ChatGPT, hence by induction, our code is also correct. We handle it in a different way but get the same result.
```


----


## Is the model produced by the LLM correct?

```
The uni gram section of the model is correct, however, the bi and tri grams are incorrect as ChatGPT didn't add N number of start tokens, rather only adds one start token causing the probabilities for phonemes that usually start the sentences to be incorrect.
```

### Is the model produced by your code different?

```
Yes, its better because we add N-1 start tokens (`["<s>"] * (max_n - 1)`), which allows probabilities in Ngrams models with n > 1 to account for probabilities of common starting phonemes.
```

----


# Other LLM code that is **incorrect**

```
N/A
```