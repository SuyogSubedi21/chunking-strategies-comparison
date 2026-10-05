from chunking import fixed_chunking, recursive_chunking, semantic_chunking
from retriever import search

with open("corpus.txt", encoding="utf-8") as f:
    corpus = f.read()

# each question and the exact text the correct chunk must contain
test_questions = [
    ("What was the net profit in 2023/24?", "NPR 4.2 billion"),
    ("How much were total deposits?", "NPR 312 billion"),
    ("What is the non-performing loan ratio?", "2.7 percent"),
    ("What is the capital adequacy ratio?", "13.1 percent"),
    ("How many mobile banking users are there?", "two million"),
    ("What cash dividend did the board propose?", "10 percent"),
    ("How many graduates did the bank hire?", "300 graduates"),
    ("What is the green lending target?", "5 percent of its loan portfolio"),
    ("How fast did home loans grow?", "18 percent"),
    ("What checks every card transaction?", "fraud monitoring system"),
]

strategies = {
    "Fixed": fixed_chunking(corpus),
    "Recursive": recursive_chunking(corpus),
    "Semantic": semantic_chunking(corpus),
}

results = {}
for name, chunks in strategies.items():
    print(f"\n===== {name} chunking: {len(chunks)} chunks =====")
    hits = 0
    for question, answer in test_questions:
        top_chunk = search(question, chunks)[0]
        found = answer in top_chunk
        hits += found
        print(f"\n{'PASS' if found else 'FAIL'}  Question:  {question}")
        print(f"      Expected:  {answer}")
        print(f"      Retrieved: {' '.join(top_chunk.split())}")
    results[name] = hits

print("\n===== Retrieval quality (answer found in top chunk) =====")
for name, hits in results.items():
    chunks = strategies[name]
    avg_size = sum(len(c) for c in chunks) // len(chunks)
    print(f"{name:<10} {hits}/{len(test_questions)} correct   {len(chunks)} chunks, avg {avg_size} chars")