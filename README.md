# Development-of-a-Long-Context-Transformer-within-RAG-for-Enhanced-Document-Understanding
API: https://console.groq.com/docs/quickstart
---

# Wikipedia RAG (Retrieval-Augmented Generation)

This project demonstrates a basic Retrieval-Augmented Generation (RAG) pipeline using Wikipedia data and OpenAI's GPT model. It retrieves relevant text from a Wikipedia article, embeds and ranks paragraphs based on similarity to a user query, and generates a response using OpenAI's GPT model with the most relevant content.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Code Walkthrough](#code-walkthrough)
- [Example Output](#example-output)
- [License](#license)

## Overview
The RAG pipeline combines Wikipedia's structured data with the reasoning power of OpenAI's GPT model to answer specific user queries. By fetching and embedding paragraphs from a Wikipedia article, the bot identifies and provides contextually relevant information to answer a question.

## Installation

To use this code in Google Colab or a local environment, install the following dependencies:
```bash
pip install -q sentence-transformers
pip install -q wikipedia-api
pip install -q numpy
pip install -q scipy
pip install -q openai
```

## Usage
1. **Run the Code**: Copy the provided code into a Python file or Google Colab cell.
2. **API Key Setup**: Ensure your OpenAI API key is saved securely, as this is required to access OpenAI's model.
3. **Query a Question**: Input a question, and the bot will fetch relevant paragraphs from Wikipedia, encode and rank them by similarity to the question, and use OpenAI's GPT model to generate an answer.

## Code Walkthrough

### 1. Setup and Installation
```python
!pip install -q sentence-transformers wikipedia-api numpy scipy
```
Install required libraries to perform text embedding, Wikipedia data retrieval, and numerical operations.

### 2. Load Embedding Model
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
```
Initialize the Sentence Transformer model to embed text into vector space for similarity comparison.

### 3. Fetch Wikipedia Text
```python
from wikipediaapi import Wikipedia
wiki = Wikipedia('en')
doc = wiki.page('Hayao_Miyazaki').text
paragraphs = doc.split('\n\n')
```
Use the Wikipedia API to fetch text about a specified topic, split it into paragraphs, and store it in a list.

### 4. Format Text with `textwrap`
```python
import textwrap

for p in paragraphs:
    print(textwrap.fill(p, width=100))
```
Format and print text for readability.

### 5. Embed Document Text
```python
docs_embed = model.encode(paragraphs, normalize_embeddings=True)
```
Encode each paragraph as a numerical vector, useful for similarity calculations.

### 6. Embed Query and Compute Similarities
```python
query = "What was Studio Ghibli's first film?"
query_embed = model.encode(query, normalize_embeddings=True)
similarities = np.dot(docs_embed, query_embed.T)
```
Encode the query and calculate similarity scores between the query and each paragraph.

### 7. Retrieve Most Relevant Paragraphs
```python
top_3_idx = np.argsort(similarities)[-3:][::-1]
most_similar_documents = [paragraphs[idx] for idx in top_3_idx]
```
Identify the top 3 most relevant paragraphs to the query.

### 8. Create Prompt for OpenAI API
```python
CONTEXT = "\n\n".join([textwrap.fill(p, width=100) for p in most_similar_documents])
prompt = f"CONTEXT: {CONTEXT}\nQUESTION: {query}"
```
Combine the top paragraphs into a context string to pass to the GPT model.

### 9. Query OpenAI's GPT Model
```python
import openai
openai.api_key = 'your_openai_api_key'

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
print(response.choices[0].message.content)
```
Use OpenAI’s GPT model to generate a response based on the provided context and question.

## Example Output
For the question:
> "What was Studio Ghibli's first film?"

The output might look like:
```
Studio Ghibli's first film was *Laputa: Castle in the Sky* (1986), directed by Hayao Miyazaki.
```

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---
