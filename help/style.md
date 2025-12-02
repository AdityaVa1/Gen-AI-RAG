# Discuss similarities and differences in coding style

---

### Comments

**Example:**

|yours|LLM|
|---|---|
|![Our Commets](https://github.com/AdityaVa1/Gen-AI-RAG/blob/main/help/images/our_comments.png)|![ChatGPT's Commets](https://github.com/AdityaVa1/Gen-AI-RAG/blob/main/help/images/gpt_comments.png)|

**Discussion:**

```
The styles are very dissimilar, chatGPT seems to ignore the assignment's description of proper documentation and adds very minimal comments that are fine for someone glancing over the code, however, if someone needs to understand and maintain the code, it would be really hard compared to our code.
```

---

### Function declarations

**Example:**

|yours|LLM|
|---|---|
|![Our Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/our_function_dec.png)|![ChatGPT's Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/gpt_function_dec.png)|

**Discussion:**

```
The functions declarations are very similar, however, chatGPT does seem to use more sensible variable names, and adds a function output type as well, which we never do.
```

---

### Organization of the JSON file

**Example:**

|yours|LLM|
|---|---|
|![Our Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/our_json.png)|![ChatGPT's Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/gpt_json.png)|

**Discussion:**

```
Very dissimilar, chatGPT tends to make much more prettier key field names and abstracts it to include as much information as cleanly as possible. Our code tends to combine all the Ngrams (uni, bi, and tri) into one key field called Ngrams and hence is much more messy.
```

---

### Print statements

**Example:**

|yours|LLM|
|---|---|
|![Our Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/our_print.png)|![ChatGPT's Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/gpt_print.png)|

**Discussion:**

```
Ironically, the print statements in both our and gpt's code are very barebones, and almost non existent. We instead have a prettier print statement for the perplexity as it was the main objective metric we were trying to minimize. This is probably due to the fact that ChatGPT as it continues to generate longer, the longer it's own context tokens grow, as the generated code/text gets added as context into its input. This would lead it to hallucinate or forget crucial details (like not printing the ppl in a detailed way for debugging).
```

---

### Presenting the perplexity results

**Example:**

|yours|LLM|
|---|---|
|![Our Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/our_print.png)|![ChatGPT's Commets](https://github.com/CMPUT-461-561/f25-asn5-b-AdityaVa1/images/gpt_print.png)|

**Discussion:**

```
I disscuss this in the previous section as well, but, we tend to display the ppl results in a slightly better way that makes it easier to debug many different models at the same time.
```

