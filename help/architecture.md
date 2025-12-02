
## Did you design your solution with classes?

```
No we didn't use any classes as our code did not need repeated functions, except for calculating the ngram counts or perplexity for each ngram, but designing a class is a bit more long winded that designing a function.
```

## Does the use of classes mean you have to type more code?

```
No, usually the point of classes is to save time and code by not having to repeat the same functions for the same type of object you are modeling. It helps you to call this class/object from anywhere in any code file. But in our case we decided that writing the functions for each ngram takes in a parameter n and based on this parameter it changes its functionality for the type of ngram.
```

---

## Functions used for model training

#### Functions you wrote with their equivalent in the LLM code (or N/A if applicable)

```
#### train.py:
get_ngrams(tokens: list, max_n: int= 3)
     - LLM equivalent: count_ngrams(utterances: List[List[str]])

get_ngram_count(input_path: Path)
     - LLM equivalent: read_utterances(path: str) and count_ngrams(utterances: List[List[str]])

save_counts(counts: dict, output_path: Path)
     - LLM equivalent: create_model_json(uni: Counter[str], bi: Counter[Tuple[str, str]], tri: Counter[Tuple[str, str, str]], vocab: List[str], n_utterances: int)
```

#### Functions the LLM wrote that have no equivalent in your code

```
build_vocab(utterances: List[List[str]])
replace_singletons_with_unk(utterances: List[List[str]], vocab: List[str])
```

#### Discussion of the similarities and differences between the two designs. 

```
The major difference is that ChatGPT implements OOV handling using UNK tokens, whereas, we used stupid backoff with a fixed weight. ChatGPT does this because it doesn't remember in its context that the data has such minimal OOV tokens (because its phonetics) that OOV using unknown tokens is useless, and most 0 probabilities are caused by unseen sequences of bi grams or tri grams.

Else it is pretty similar, just the order of operations feels more streamlined than our code. Like how ChatGPT abstracts the code for calculating the probability for uni. bi, and tri grams, whereas, we fitted it all into one function.

The reason for the similarities is that functions like loading data and calculating probability are fairly independent and were probably seen individually a lot of time during training.

Another thing that ChatGPT does that we don't is that it has a build_vocab function, it does this because it assumes the vocabulary will be used at some point of the future code. It is inclined to think this because most of the NLP code online tackling ngram models would also contain the vocabulary dictionary because it is a useful and common tool to have.

We don't save a vocabulary dictionary because we don't handle OOV tokens using UNK tokens.
```

#### Which design is better?

```
Definitely chatGPT's design is better because it follows coding and design principles from all the programming books written and published (from libGen, stolen). It is readable and commented just enough to make sense, the names of the variables, parameters, and arguments are sensible, unlike ours.

Especially having a vocabulary and building functions on top of that is cleaner and more understandable instead of coding the functions within one function.
```

---

## Functions used for calculating perplexity

#### Functions you wrote with their equivalent in the LLM code (or N/A if applicable)

```
#### eval.py:
get_prob(ngram: str, n: int, ngram_counts: dict, laplace: bool, weight: float = 0.4)
     - LLM equivalent: p_unigram(w: str, model: Dict[str, Any]) and p_bigram(w1: str, w2: str, model: Dict[str, Any], laplace: bool) and p_trigram(w1: str, w2: str, w3: str, model: Dict[str, Any], laplace: bool)

get_tokens(line: str, n: int)
     - LLM equivalent: start of compute_ppl(model_type: str, model: Dict[str, Any], utterances: List[List[str]], laplace: bool, vocab: set[str])

main(model_type: str, model_path: Path, data_path: Path, laplace: bool)
     - LLM equivalent: compute_ppl(model_type: str, model: Dict[str, Any], utterances: List[List[str]], laplace: bool, vocab: set[str])
```

#### Functions the LLM wrote that have no equivalent in your code

```
convert_to_json(counter: Counter)
map_oov(tokens: List[str], vocab: set[str])
```

#### Discussion of the similarities and differences between the two designs. 

```
The same thing as before where ChatGPT uses UNK as an implementation to handle OOV tokens. It also separates the methods to calculate the probabilities of each ngram, whereas we do it in one function.
```

#### Which design is better?

```
ChatGPT has a cleaner and more concise implementation, while, ours tries to implement all the probability calculation in one function along with the fact that our model.json has all uni, bi, and tri grams all under one document. These overlapping and more complicated processes are broken down further into simpler tasks by chatGPT.
```


---

## Functions used for printing results

#### Functions you wrote with their equivalent in the LLM code (or N/A if applicable)

```
print(f"{model_type}{smoothing_str} perplexity: {ppl:.4f}")
```

#### Functions the LLM wrote that have no equivalent in your code

```
N/A
```

#### Discussion of the similarities and differences between the two designs. 

```
ChatGPT has a much more simplistic version of our print statement where it just prints the perplexity, which makes it harder to debug
```

#### Which design is better?

```
Our is better in this case.
```
