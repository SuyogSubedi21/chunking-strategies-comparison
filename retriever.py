import math
import re
from collections import Counter

from chunking import STOP_WORDS


def tokenize(text):
    words = re.findall(r"[a-z0-9.]+", text.lower())
    return [w.strip(".")[:5] for w in words if w.strip(".") and w.strip(".") not in STOP_WORDS]


def search(question, chunks, top_k=1):
    """Score every chunk with TF-IDF on the question's words and return the best ones."""
    chunk_words = [Counter(tokenize(c)) for c in chunks]
    n = len(chunks)

    scores = []
    for i, words in enumerate(chunk_words):
        score = 0
        for word in set(tokenize(question)):
            chunks_with_word = sum(1 for cw in chunk_words if word in cw)
            idf = math.log((n + 1) / (chunks_with_word + 1)) + 1
            score += words[word] * idf
        scores.append((score / math.sqrt(sum(words.values()) or 1), i))  # don't favour long chunks

    scores.sort(reverse=True)
    return [chunks[i] for _, i in scores[:top_k]]