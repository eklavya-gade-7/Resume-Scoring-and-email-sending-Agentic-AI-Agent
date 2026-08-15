from sentence_transformers import CrossEncoder
import numpy as np


# Swapped from cross-encoder/ms-marco-MiniLM-L6-v2. Ettin scores 0.5779 on
# MTEB eng v2 against ms-marco's 0.5066, and it even beats the 568M
# bge-reranker-v2-m3 (0.5526) at a seventeenth of the size. On this resume it
# lifted a real computer-vision project from 8th place to 3rd, which is exactly
# the mistake that made the CV requirement fail earlier.
#
# Raw logits, no sigmoid. Sigmoid is monotonic so it never changed any ranking,
# and MMR min-max normalises afterwards regardless - but it squashed every score
# into 0.993..0.999, which made the evidence report useless for spotting that
# the reranker was not discriminating at all. Raw scores show the real spread.
reranker_model = CrossEncoder("cross-encoder/ettin-reranker-32m-v1")


def rerank_chunks(subquery, top_indices, chunks):

    pairs = []

    for index in top_indices:
        pairs.append(
            (
                subquery,
                chunks[index]["text"]
            )
        )

    reranker_scores = reranker_model.predict(pairs)

    order = np.argsort(reranker_scores)[::-1]

    reranked_indices = []
    reranked_scores = []

    for position in order:
        reranked_indices.append(top_indices[position])
        reranked_scores.append(reranker_scores[position])

    return np.array(reranked_indices), np.array(reranked_scores)
