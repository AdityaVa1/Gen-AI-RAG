## Most similar functionality

### Your code

```Python
def get_ngrams(tokens: list, max_n: int= 3):
    """
    Get all the n-gram (1 to max_n) from a list of tokens
    with the appropriate start and end padding.

    Args:
        tokens (list[str]): List of words in a sentence
        max_n (int, optional): Highest n-gram size. Defaults to 3.

    Returns:
        list[str]: List of all n-grams
    """
    
    # only need a single </s>, bc once we predict it we are done
    padded_tokens = ["<s>"] * (max_n - 1) + tokens + ["</s>"]

    ngrams = [ # list comprehension for efficeny
        " ".join(padded_tokens[i:i+n])
        for n in range(1, max_n + 1)
        for i in range(max_n - n, len(padded_tokens) - (n - 1))
    ]
    
    # we need the count of n - 1 starting tokens to compute n-grams 
    # (ex. Pr(<s> w) =  count(<s> w) / count(<s>))
    ngrams.extend([' '.join(['<s>'] * i) for i in range(1, max_n)])
    
    return ngrams

def get_ngram_count(input_path: Path):
    """
    Reads and store all the n-gram count up to max_n (3)
    along with the total token and unique token in the file

    Args:
        input_path (Path): Path to the text file who's n-gram is counted

    Returns:
        dict: Dictionary with the total token, unique token and ngram counts
    """
    ngram_count = Counter()
    total_tokens = 0
    unique_tokens = {'</s>'} # we can predict </s> but not <s>
    
    # read file line by line
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            ngrams = get_ngrams(tokens)
            ngram_count.update(ngrams) # count of all the n-grams
            total_tokens += len(tokens) + 1 # incldue </s> but not <s>
            unique_tokens.update(tokens) # store all unique words
                      
    return {
        "total_tokens": total_tokens,
        "unique_tokens": len(unique_tokens),
        "ngrams": ngram_count,
    }

def save_counts(counts: dict, output_path: Path):
    """
    Save the dict by storing the n-gram count to a JSON file

    Args:
        counts (dict): Dict with the total token, unique token, and n-gram counts
        output_path (Path): Path to save the JSON file with total token, unique token, and n-gram counts
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)
```

### LLM code

```Python
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

def read_utterances(path: str) -> List[List[str]]:
    """Reads a data file. Each line is one space-separated utterance."""
    utts: List[List[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(line.split())
    return utts

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
```

### Output of `diff`

```
diff --git a/f1.py b/f2.py
index 13c83e1..f27752f 100644
--- a/f1.py
+++ b/f2.py
@@ -1,68 +1,67 @@
def [-get_ngrams(tokens: list, max_n: int= 3):-]
[-    """-]
[-    Get all the n-gram (1 to max_n) from a list of tokens-]
[-    with the appropriate start and end padding.-]{+count_ngrams(+}
{+    utterances: List[List[str]]+}
{+) -> Tuple[+}
{+    Counter[str],+}
{+    Counter[Tuple[str, str]],+}
{+    Counter[Tuple[str, str, str]]+}
{+]:+}
{+    uni: Counter[str] = Counter()+}
{+    bi: Counter[Tuple[str, str]] = Counter()+}
{+    tri: Counter[Tuple[str, str, str]] = Counter()+}

    [-Args:-]
[-        tokens (list[str]): List of words-]{+for utt+} in [-a sentence-]
[-        max_n (int, optional): Highest n-gram size. Defaults to 3.-]{+utterances:+}
{+        seq = [START] + utt + [END]+}

        [-Returns:-]
[-        list[str]: List of all n-grams-]
[-    """-]
[-    -]
[-    # only need a single </s>, bc once we predict it we are done-]
[-    padded_tokens = ["<s>"] * (max_n - 1) + tokens + ["</s>"]-]{+uni.update(seq)+}

[-ngrams = [ # list comprehension for efficeny-]
[-        " ".join(padded_tokens[i:i+n])-]
[-        for n in range(1, max_n + 1)-]        for i in [-range(max_n - n, len(padded_tokens) - (n - 1))-]
[-    ]-]
[-    -]
[-    # we need the count of n-]{+range(len(seq)+} - {+1):+}
{+            bi[(seq[i], seq[i + 1])] +=+} 1[-starting tokens to compute n-grams -]
[-    # (ex. Pr(<s> w) =  count(<s> w) / count(<s>))-]
[-    ngrams.extend([' '.join(['<s>'] * i) for i in range(1, max_n)])-]
[-    -]
[-    return ngrams-]

        [-def get_ngram_count(input_path: Path):-]
[-    """-]
[-    Reads and store all the n-gram count up to max_n (3)-]
[-    along with the total token and unique token-]{+for i+} in [-the file-]{+range(len(seq) - 2):+}
{+            tri[(seq[i], seq[i + 1], seq[i + 2])] += 1+}

    [-Args:-]
[-        input_path (Path): Path to the text file who's n-gram is counted-]{+return uni, bi, tri+}

[-Returns:-]
[-        dict: Dictionary with the total token, unique token and ngram counts-]
[-    """-]
[-    ngram_count = Counter()-]
[-    total_tokens = 0-]
[-    unique_tokens = {'</s>'} # we can predict </s> but not <s>-]
[-    -]
[-    # read file line by-]{+def read_utterances(path: str) -> List[List[str]]:+}
{+    """Reads a data file. Each+} line {+is one space-separated utterance."""+}
{+    utts: List[List[str]] = []+}
    with [-input_path.open("r", encoding="utf-8")-]{+open(path, "r")+} as f:
        for line in f:
            [-tokens-]{+line+} = [-line.strip().split()-]
[-            ngrams = get_ngrams(tokens)-]
[-            ngram_count.update(ngrams) # count of all the n-grams-]
[-            total_tokens += len(tokens) + 1 # incldue </s> but not <s>-]
[-            unique_tokens.update(tokens) # store all unique words-]{+line.strip()+}
{+            if line:+}
{+                utts.append(line.split())+}
    return [-{-]
[-        "total_tokens": total_tokens,-]
[-        "unique_tokens": len(unique_tokens),-]
[-        "ngrams": ngram_count,-]
[-    }-]{+utts+}

{+def create_model_json(+}
{+    uni: Counter[str],+}
{+    bi: Counter[Tuple[str, str]],+}
{+    tri: Counter[Tuple[str, str, str]],+}
{+    vocab: List[str],+}
{+    n_utterances: int+}
{+) -> Dict[str, Any]:+}

    [-def save_counts(counts: dict, output_path: Path):-]
[-    """-]
[-    Save the dict by storing the n-gram count to a JSON file-]{+V = len(vocab)+}

    [-Args:-]{+return {+}
{+        "meta": {+}
{+            "description": "n-gram+} counts [-(dict): Dict with the total token, unique token, and n-gram counts-]
[-        output_path (Path): Path to save the JSON file with total token, unique token, and n-gram counts-]
[-    """-]
[-    with open(output_path, "w", encoding="utf-8") as f:-]
[-        json.dump(counts, f, ensure_ascii=False, indent=2)-]{+without smoothing",+}
{+            "bos": START,+}
{+            "eos": END,+}
{+            "unk": UNK,+}
{+            "n_sentences": n_utterances,+}
{+            "unigram_total": sum(uni.values()),+}
{+            "vocab_size_conditioning": V+}
{+        },+}
{+        "unigram": {+}
{+            "order": 1,+}
{+            "counts": convert_to_json(uni)+}
{+        },+}
{+        "bigram": {+}
{+            "order": 2,+}
{+            "counts": convert_to_json(bi)+}
{+        },+}
{+        "trigram": {+}
{+            "order": 3,+}
{+            "counts": convert_to_json(tri)+}
{+        }+}
{+    }+}
```

## Discussion

### What is most similar?

```
The logic and order of steps is the most similar. ChatGPT and our code does the work in the same general order of steps. We first get the Ngrams from the uttrances, then get the counts, save the counts, load the counts, get the test uttrances, get the probability (either uni, bi, or tri), if the probability doesn't exist then we use stupid backoff and chatGPT uses inf probabilities, and then get the perplexity.
```

### What is most different?

```
The style, variables names, and some major logic. The style of ChatGPT is much more cleaner and properly abstracted to functions and their tasks, with more sensible variable names and function names. The most different logic is the difference of handling 0 probabilities when dealing with OOV tokens in the test set, we use stupid backoff to fallback to a simpler model, while ChatGPT just gives up and returns an inf probability. 
```
