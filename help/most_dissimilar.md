## Most dissimilar functionality

### Your code

```Python
def get_prob(
    ngram: str, 
    n: int, 
    ngram_counts: dict, 
    laplace: bool, 
    weight: float = 0.4):
    """
    Recursively calculating the probabilty of the n-gram by
    direct counts or stupid backoff

    Args:
        ngram (str): n-gram who's probabilty needs to be found
        n (int): n-gram size (1 = unigram, 2 = bigram, 3 = trigram)
        ngram_counts (dict): The saved n-gram count
        laplace (bool): Flag telling if to use Laplace smoothing
        weight (float, optional): Backoff weight. Defaults to 0.4.

    Returns:
        float: Calculated probability
    
    Note:
        The weight of 0.4 was recommened by the textbook
    """
    
    if n == 0: # zerogram (base case for backoff recursion)
        return 1 / ngram_counts['total_tokens']
    
    # count of the full n-gram (joint probability)
    joint_count = ngram_counts['ngrams'].get(ngram, 0)
    
    # context count (conditional probability)
    if n == 1:
        context_count = ngram_counts['total_tokens']
    else:
        context = " ".join(ngram.split()[:-1])
        context_count = ngram_counts['ngrams'].get(context, 0)
    
    # Using laplace for smoothing if needed
    if laplace:
        joint_count += 1
        context_count += ngram_counts['unique_tokens']  
    
    # n-gram or context OOV, we backoff to lower n-gram
    if joint_count == 0 or context_count == 0:
        # stupid backoff (ie. use fixed weight rather than calculating a discount that ensures total proability mass is 1.0)
        n_minus_one_gram = " ".join(ngram.split()[1:])
        prob = weight * get_prob(n_minus_one_gram, n-1, ngram_counts, laplace)
    else:
        prob = joint_count / context_count
        
    return prob

def get_tokens(line: str, n: int):
    """
    Tokenize given line and add start and end
    tokens for n-grams

    Args:
        line (str): Line that needs to be tokenized
        n (int): Size of n-gram

    Returns:
        list[str]: list of token with the start and end padding
    """
    # only need a single </s>, bc once we predict it we are done
    padded_line =  "<s> " * (n - 1) + line + " </s>"
    return padded_line.split()

def main(
    model_type: str, 
    model_path: Path, 
    data_path: Path, 
    laplace: bool):
    """
    Calculating perplexity using a n-gram model

    Args:
        model_type (str): "unigram", "bigram", or "trigram"
        model_path (Path): Path to JSON with the n-gram counts
        data_path (Path): Path to the file to evaluate
        laplace (bool): flag telling wheather to laplace smoothing
    """
    
    # load the model from the JSON file
    with model_path.open("r", encoding="utf-8") as f:
        ngram_counts = json.load(f)

    # model name to the n-gram size
    model_type_to_n = {
        "unigram": 1,
        "bigram": 2,
        "trigram": 3
    }
    n = model_type_to_n[model_type]
    
    total_log = 0.0
    total_ngrams = 0
    
    # read file line by line
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            # convert a line to token and make all the n-grams
            tokens = get_tokens(line, n)
            ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - (n - 1))]
            total_ngrams += len(ngrams)

            # add the log probabilty of each n-gram to total
            for ngram in ngrams:
                print(ngram)
                prob = get_prob(ngram, n, ngram_counts, laplace)
                total_log += math.log(prob)
    
    # Get average log probability and the perplexity     
    log_avg = total_log / total_ngrams
    ppl = math.exp(-log_avg)
    
    smoothing = 'laplace' if laplace else 'unsmoothed'
    smoothing_str = ' ' + smoothing if model_type != 'unigram' else ""
    print(f"{model_type}{smoothing_str} perplexity: {ppl:.4f}")
```

### LLM code

```Python
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
```

### Output of `diff`

```
diff --git a/f1.py b/f2.py
index 523f838..0ee85db 100644
--- a/f1.py
+++ b/f2.py
@@ -1,119 +1,77 @@
def [-get_prob(-]
[-    ngram:-]{+p_unigram(w:+} str, [-n: int, -]
[-    ngram_counts: dict,-]{+model: Dict[str, Any]) -> float:+}
{+    uni = model["unigram"]+}
{+    total = model["meta"]["unigram_total"]+}
{+    return uni.get(w, 0) / total+}


{+def p_bigram(w1: str, w2: str, model: Dict[str, Any],+} laplace: [-bool, -]
[-    weight: float-]{+bool) -> float:+}
{+    bi+} = [-0.4):-]
[-    """-]
[-    Recursively calculating the probabilty of the n-gram by-]
[-    direct counts or stupid backoff-]

[-    Args:-]
[-        ngram (str): n-gram who's probabilty needs to be found-]
[-        n (int): n-gram size (1-]{+model["bigram"]+}
{+    uni+} = [-unigram, 2-]{+model["unigram"]+}
{+    V+} = [-bigram, 3-]{+model["meta"]["vocab_size_conditioning"]+}

{+    num = bi.get((w1, w2), 0)+}
{+    den+} = [-trigram)-]
[-        ngram_counts (dict): The saved n-gram count-]
[-        laplace (bool): Flag telling-]{+uni.get(w1, 0)+}

    if [-to use Laplace smoothing-]
[-        weight (float, optional): Backoff weight. Defaults to 0.4.-]

[-    Returns:-]
[-        float: Calculated probability-]
[-    -]
[-    Note:-]
[-        The weight of 0.4 was recommened by the textbook-]
[-    """-]{+laplace:+}
{+        return (num + 1) / (den + V)+}

    if [-n-]{+den+} == 0:[-# zerogram (base case for backoff recursion)-]
        return [-1-]{+0.0+}
{+    return num+} / [-ngram_counts['total_tokens']-]
[-    -]
[-    # count of the full n-gram (joint probability)-]
[-    joint_count-]{+den+}


{+def p_trigram(w1: str, w2: str, w3: str, model: Dict[str, Any], laplace: bool) -> float:+}
{+    tri+} = [-ngram_counts['ngrams'].get(ngram, 0)-]
[-    -]
[-    # context count (conditional probability)-]
[-    if n == 1:-]
[-        context_count-]{+model["trigram"]+}
{+    bi+} = [-ngram_counts['total_tokens']-]
[-    else:-]
[-        context-]{+model["bigram"]+}
{+    V+} = [-" ".join(ngram.split()[:-1])-]
[-        context_count-]{+model["meta"]["vocab_size_conditioning"]+}

{+    num+} = [-ngram_counts['ngrams'].get(context,-]{+tri.get((w1, w2, w3), 0)+}
{+    den = bi.get((w1, w2),+} 0)[-# Using laplace for smoothing if needed-]

    if laplace:
        [-joint_count += 1-]
[-        context_count += ngram_counts['unique_tokens']  -]
[-    -]
[-    # n-gram or context OOV, we backoff to lower n-gram-]{+return (num + 1) / (den + V)+}

    if [-joint_count == 0 or context_count-]{+den+} == 0:[-# stupid backoff (ie. use fixed weight rather than calculating a discount that ensures total proability mass is 1.0)-]
[-        n_minus_one_gram = " ".join(ngram.split()[1:])-]
[-        prob = weight * get_prob(n_minus_one_gram, n-1, ngram_counts, laplace)-]
[-    else:-]
[-        prob = joint_count / context_count-]
        return [-prob-]

[-def get_tokens(line: str, n: int):-]
[-    """-]
[-    Tokenize given line and add start and end-]
[-    tokens for n-grams-]

[-    Args:-]
[-        line (str): Line that needs to be tokenized-]
[-        n (int): Size of n-gram-]

[-    Returns:-]
[-        list[str]: list of token with the start and end padding-]
[-    """-]
[-    # only need a single </s>, bc once we predict it we are done-]
[-    padded_line =  "<s> " * (n - 1) + line + " </s>"-]{+0.0+}
    return [-padded_line.split()-]{+num / den+}

def [-main(-]{+compute_ppl(+}
    model_type: str,
    [-model_path: Path, -]
[-    data_path: Path,-]{+model: Dict[str, Any],+}
{+    utterances: List[List[str]],+}
    laplace: [-bool):-]
[-    """-]
[-    Calculating perplexity using a n-gram model-]

[-    Args:-]
[-        model_type (str): "unigram", "bigram", or "trigram"-]
[-        model_path (Path): Path to JSON with the n-gram counts-]
[-        data_path (Path): Path to the file to evaluate-]
[-        laplace (bool): flag telling wheather to laplace smoothing-]
[-    """-]
[-    -]
[-    # load the model from the JSON file-]
[-    with model_path.open("r", encoding="utf-8") as f:-]
[-        ngram_counts = json.load(f)-]

[-    # model name to the n-gram size-]
[-    model_type_to_n = {-]
[-        "unigram": 1,-]
[-        "bigram": 2,-]
[-        "trigram": 3-]
[-    }-]
[-    n = model_type_to_n[model_type]-]
[-    -]
[-    total_log-]{+bool,+}
{+    vocab: set[str]+}
{+) -> float:+}

{+    log_prob_sum+} = 0.0
    [-total_ngrams-]{+N+} = 0[-# read file line by line-]
[-    with data_path.open("r", encoding="utf-8") as f:-]

    for [-line-]{+utt+} in [-f:-]
[-            # convert a line to token and make all the n-grams-]
[-            tokens-]{+utterances:+}
{+        utt+} = [-get_tokens(line, n)-]
[-            ngrams-]{+map_oov(utt, vocab)+}
{+        seq+} = [-[" ".join(tokens[i:i+n])-]{+[START] + utt + [END]+}

{+        if model_type == "unigram":+}
            for [-i-]{+w+} in [-range(len(tokens) - (n - 1))]-]
[-            total_ngrams-]{+seq:+}
{+                p = p_unigram(w, model)+}
{+                if p == 0:+}
{+                    return float("inf")+}
{+                log_prob_sum += math.log(p)+}
{+                N+} += [-len(ngrams)-]

[-            # add the log probabilty of each n-gram to total-]{+1+}

{+        elif model_type == "bigram":+}
            for [-ngram-]{+i+} in [-ngrams:-]
[-                print(ngram)-]
[-                prob-]{+range(len(seq) - 1):+}
{+                p+} = [-get_prob(ngram, n, ngram_counts,-]{+p_bigram(seq[i], seq[i + 1], model,+} laplace)[-total_log += math.log(prob)-]
[-    -]
[-    # Get average log probability and the perplexity     -]
[-    log_avg = total_log / total_ngrams-]
[-    ppl = math.exp(-log_avg)-]
[-    -]
[-    smoothing = 'laplace'-]
                if [-laplace else 'unsmoothed'-]
[-    smoothing_str-]{+p == 0:+}
{+                    return float("inf")+}
{+                log_prob_sum += math.log(p)+}
{+                N += 1+}

{+        elif model_type == "trigram":+}
{+            for i in range(len(seq) - 2):+}
{+                p+} = [-' '-]{+p_trigram(seq[i], seq[i + 1], seq[i+} + [-smoothing-]{+2], model, laplace)+}
                if [-model_type != 'unigram' else ""-]
[-    print(f"{model_type}{smoothing_str} perplexity: {ppl:.4f}")-]{+p == 0:+}
{+                    return float("inf")+}
{+                log_prob_sum += math.log(p)+}
{+                N += 1+}

{+    return math.exp(-log_prob_sum / N)+}
```

## Discussion

### What is most similar?

```
The general logic and order of steps is the most similar.
```

### What is most different?

```
The most different part of the code is the calculation of the uni, bi, and trigram probabilities. This is because ChatGPT uses separate functions to calculate the ngram probabilities, but we make a function that works with any n gram.
```
