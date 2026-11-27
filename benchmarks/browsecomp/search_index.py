"""Small deterministic BM25 index for one frozen query's candidate documents."""

import math
import re
from collections import Counter, defaultdict

WORDS = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and are as at be by for from had has have in is it of on or that the to was were with who what where when which how its their this".split())


def terms(text):
    return [word for word in WORDS.findall(text.lower()) if len(word) > 1 and word not in STOP]


class SearchIndex:
    def __init__(self, row):
        unique = {}
        for group in ("evidence_docs", "gold_docs", "negative_docs"):
            for document in row[group]:
                unique.setdefault(document["docid"], document["text"])
        self.chunks = {}
        self.postings = defaultdict(list)
        self.lengths = []
        for docid, text in unique.items():
            words = text.split()
            for start in range(0, len(words), 160):
                window = " ".join(words[start:start + 200])
                if not window:
                    continue
                chunk_id = f"c{len(self.chunks)}"
                self.chunks[chunk_id] = {"docid": docid, "text": window}
                tokens = terms(window)
                index = len(self.lengths)
                self.lengths.append(len(tokens))
                for token, count in Counter(tokens).items():
                    self.postings[token].append((index, count))
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1

    def search(self, query, limit=5):
        if limit < 1 or limit > 5:
            return {"error": "limit must be 1..5"}
        scores = defaultdict(float)
        n = len(self.lengths)
        for token in set(terms(query)):
            posting = self.postings.get(token, [])
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, frequency in posting:
                length = self.lengths[index]
                scores[index] += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / self.average_length))
        hits = []
        seen_docs = Counter()
        for index, value in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0])):
            chunk_id = f"c{index}"
            chunk = self.chunks[chunk_id]
            if seen_docs[chunk["docid"]] >= 2:
                continue
            seen_docs[chunk["docid"]] += 1
            hits.append({"chunk_id": chunk_id, "docid": chunk["docid"],
                         "snippet": chunk["text"][:500], "score": round(value, 3)})
            if len(hits) == limit:
                break
        return {"hits": hits}

    def read(self, chunk_ids):
        if not chunk_ids or len(chunk_ids) > 5:
            return {"error": "read one to five chunk IDs"}
        unknown = [chunk_id for chunk_id in chunk_ids if chunk_id not in self.chunks]
        if unknown:
            return {"error": f"unknown chunk IDs: {unknown}"}
        return {chunk_id: self.chunks[chunk_id] for chunk_id in chunk_ids}
