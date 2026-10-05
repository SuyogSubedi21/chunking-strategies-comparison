import re

STOP_WORDS = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "was", "were",
              "are", "by", "with", "its", "it", "this", "that", "at", "as", "be", "will", "also", "has",
              "bank", "npr", "billion", "percent", "year"}  # words that appear in every topic


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# 1. Fixed chunking: cut every `size` characters
def fixed_chunking(text, size=200, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# 2. Recursive chunking: split by paragraphs, then lines, then sentences, then words,
#    only going to a smaller separator when a piece is still too big
def recursive_chunking(text, size=200, separators=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text.strip()] if text.strip() else []
    if not separators:                       # nothing left to split on, cut by characters
        return [text[i:i + size] for i in range(0, len(text), size)]

    sep, smaller_seps = separators[0], separators[1:]
    chunks = []
    current = ""
    for piece in text.split(sep):
        if len(piece) > size:                # piece too big, split it with the next separator
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(recursive_chunking(piece, size, smaller_seps))
        elif len(current) + len(sep) + len(piece) <= size:
            current = current + sep + piece if current else piece   # still fits, keep adding
        else:
            chunks.append(current.strip())   # full, save it and start a new one
            current = piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


# 3. Semantic chunking: start a new chunk when a sentence shares no words with the previous one
def keywords(sentence):
    words = re.findall(r"[a-z]+", sentence.lower())
    return {w[:5] for w in words if w not in STOP_WORDS and len(w) > 2}  # first 5 letters: "loans" = "loan"


def semantic_chunking(text, max_size=400):
    sentences = split_sentences(text)
    chunks = []
    current = [sentences[0]]
    for prev, sentence in zip(sentences, sentences[1:]):
        shared = keywords(prev) & keywords(sentence)
        too_big = len(" ".join(current)) + len(sentence) > max_size
        if not shared or too_big:            # topic changed (or chunk full), start a new chunk
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)
    chunks.append(" ".join(current))
    return chunks