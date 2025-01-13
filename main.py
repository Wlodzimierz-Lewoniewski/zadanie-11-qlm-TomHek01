import math
from collections import Counter, defaultdict

def tokenize(text):

    return text.lower().split()

def calculate_probabilities(documents, query, lambda_param=0.5):

    tokenized_docs = [tokenize(doc) for doc in documents]
    query_tokens = tokenize(query)

    doc_term_counts = [Counter(doc) for doc in tokenized_docs]
    corpus_term_count = Counter(term for doc in tokenized_docs for term in doc)
    corpus_size = sum(corpus_term_count.values())

    doc_probabilities = []

    for idx, doc_count in enumerate(doc_term_counts):
        doc_size = sum(doc_count.values())
        log_prob = 0  

        for term in query_tokens:
            
            doc_term_freq = doc_count.get(term, 0)
            
            corpus_term_freq = corpus_term_count.get(term, 0)

            smoothed_prob = (
                lambda_param * (doc_term_freq / doc_size) +
                (1 - lambda_param) * (corpus_term_freq / corpus_size)
            )

            if smoothed_prob > 0:
                log_prob += math.log(smoothed_prob)

        doc_probabilities.append((idx, log_prob))

    doc_probabilities.sort(key=lambda x: (round(x[1], 10), x[0]), reverse=True)

    return doc_probabilities

def main():

    n = int(input())
    documents = []

    for _ in range(n):
        documents.append(input().strip())

    query = input().strip()

    lambda_param = 0.5
    ranked_docs = calculate_probabilities(documents, query, lambda_param)

    result = [idx for idx, _ in ranked_docs]
    print(result)

if __name__ == "__main__":
    main()
