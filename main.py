import math
from collections import defaultdict

def split_text(text):

    return text.lower().replace('.', '').split()

def compute_probabilities(documents, search_query, param=0.5):

    tokenized_documents = [split_text(doc) for doc in documents]
    tokenized_query = split_text(search_query)

    corpus_frequency = defaultdict(int)
    for doc in tokenized_documents:
        for term in doc:
            corpus_frequency[term] += 1
    total_terms = sum(corpus_frequency.values())

    results = []

    for index, doc in enumerate(tokenized_documents):
        doc_length = len(doc)
        term_frequency = defaultdict(int)

        for term in doc:
            term_frequency[term] += 1

        log_probability = 0
        for term in tokenized_query:
            term_in_doc_freq = term_frequency.get(term, 0)
            term_in_corpus_freq = corpus_frequency.get(term, 0)

            smoothed_probability = (
                param * (term_in_doc_freq / doc_length if doc_length > 0 else 0) +
                (1 - param) * (term_in_corpus_freq / total_terms if total_terms > 0 else 0)
            )

            if smoothed_probability > 0:
                log_probability += math.log(smoothed_probability)

        results.append((index, log_probability))

    results.sort(key=lambda x: (-x[1], x[0]))

    return [index for index, _ in results]

num_docs = int(input())
documents_list = [input().strip() for _ in range(num_docs)]
query_input = input().strip()

x = compute_probabilities(documents_list, query_input)

print(x)

