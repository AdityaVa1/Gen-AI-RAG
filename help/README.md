## Assignment 3 URL

Complete your assignment 3 URL:

https://github.com/CMPUT-461-561/f25-asn3-loganwoudstra



---

## Choice of LLM to compare against:

```
ChatGPT
```

### Brief comments

```
Claude seems to generate much more comments and documentations than chatGPT.
Claude also seems to generate many more pretty print statements than chatGPT.
Both ChatGPT and Claude use very similar input parameters and naming conventions.

### In the train.py files:
- Number of functions:
     - ChatGPT: 7
     - Claude: 7
     - Our Code: 4 (+ 2 from data loading python file)
- Differences: 
We split our data loading into a separate python file. We first open the file and go line by line to get a list of ngrams in the line with boundary tokens. Then we use a Counter to add or update the count of the list of ngram counts for each ngram. Then we save these counts to a json file. 

The closest is ChatGPT because it handles all ngram counts in one function like we do. Whereas, Claude uses a separate function for each ngram. However, ChatGPT also handles OOV tokens by replacing singletons with <unk> tokens. We don't do this, and neither does Claude.

### In the eval.py files:
- Number of functions:
     - ChatGPT: 8
     - Claude: 6
     - Our Code: 3
- Differences:
ChatGPT uses the OOV handling technique of replacing unseen tokens with <unk> tokens. Claude instead adds a very small probability for an unseen context. Instead, we use stupid backoff to compute the probabilities of unseen context or joint probabilities (0 prob). Claude once again uses 3 functions, one for each ngram, to compute the perplexity. ChatGPT this time also does the same as Claude. Whereas, we incorporate the ngram logic in a boilerplate function which takes in the size of the ngram within the function arguments.
```

---



## "Copy-and-paste" execution instructions to run the **LLM code** after your changes

```
python src/chatgpt/train.py data/tiny\\\_training.txt output/chatgpt/model.json
python src/chatgpt/eval.py bigram output/chatgpt/model.json data/tiny\\\_dev.txt --laplace
python src/chatgpt/eval.py trigram output/chatgpt/model.json data/tiny\\\_dev.txt --laplace
python src/chatgpt/eval.py trigram output/chatgpt/model.json data/tiny\\\_dev.txt
python src/chatgpt/eval.py bigram output/chatgpt/model.json data/tiny\\\_dev.txt
python src/chatgpt/eval.py unigram output/chatgpt/model.json data/tiny\\\_dev.txt
python src/chatgpt/eval.py bigram output/chatgpt/model.json data/tiny\\\_training.txt --laplace
python src/chatgpt/eval.py trigram output/chatgpt/model.json data/tiny\\\_training.txt --laplace
python src/chatgpt/eval.py trigram output/chatgpt/model.json data/tiny\\\_training.txt
python src/chatgpt/eval.py bigram output/chatgpt/model.json data/tiny\\\_training.txt
python src/chatgpt/eval.py unigram output/chatgpt/model.json data/tiny\\\_training.txt
```



---

## YOUR Evaluation **copied** from what you submitted in assignment 3

|Model           | Smoothing  | Training set PPL | Dev set PPL |
|----------------|----------- | ---------------- | ----------- |
|unigram         | -          |     24.9584      |    24.9757  |
|bigram          | unsmoothed |     9.5720       |     9.5942  |
|bigram          | Laplace    |      9.5802      |     9.6030  |
|trigram         | unsmoothed |      5.2685      |    5.2824   |
|trigram         | Laplace    |       5.3656     |      5.3780 |



## LLM evaluation

|Model           | Smoothing  | Training set PPL | Dev set PPL |
|----------------|----------- | ---------------- | ----------- |
|unigram         | -          |      21.6032     |   21.6214   |
|bigram          | unsmoothed |       9.5720     |      inf    |
|bigram          | Laplace    |       9.5802     |    9.6030   |
|trigram         | unsmoothed |       3.1109     |      inf    |
|trigram         | Laplace    |       3.1877     |    3.1979   |

