from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def normalize_scores(scores):

    scores = np.array(scores, dtype=float)

    if len(scores) == 0:
        return scores

    minimum = np.min(scores)
    maximum = np.max(scores)

    if np.isclose(maximum, minimum):
        return np.ones(len(scores))

    normalized_scores = (scores - minimum) / (maximum - minimum)

    return normalized_scores


def maximal_marginal_relevance(reranked_indices, reranker_scores, vector_db,
                               final_k=5, lambda_mult=0.8):
    # Tried and rejected: force-keeping the best keyword-retrieved chunk here,
    # to cover the cross-encoder under-rating bare skills lines. Measured over
    # 11 requirements it made things worse, 9 correct down to 8 - the seeded
    # chunk displaced a genuinely better one on the computer-vision requirement,
    # and it did not even rescue the case it was written for.

    reranked_indices = np.array(reranked_indices)
    reranker_scores = np.array(reranker_scores, dtype=float)

    if len(reranked_indices) == 0:
        return np.array([]), np.array([]), np.array([])

    if len(reranked_indices) != len(reranker_scores):
        raise ValueError("reranked_indices and reranker_scores must have the same length")

    final_k = min(final_k, len(reranked_indices))

    # Centre the vectors before measuring redundancy. bge-small embeddings are
    # anisotropic - every vector leans toward one shared direction, so on this
    # resume any two chunks scored between 0.397 and 0.812 against each other,
    # mean 0.578. With the redundancy term squashed into that narrow band the
    # penalty was nearly the same for every candidate, which made MMR collapse
    # into "just take the reranker's order". Subtracting the mean vector spreads
    # the same comparisons over -0.330 to 0.603, and MMR starts doing its job:
    # measured over 12 criteria it changed the selected evidence 6 times instead
    # of 4. Ranking is unaffected - this only feeds the diversity penalty.
    vector_db = np.asarray(vector_db, dtype=float)
    vector_db = vector_db - vector_db.mean(axis=0)

    normalized_scores = normalize_scores(reranker_scores)

    first_position = np.argmax(normalized_scores)

    selected_positions = [first_position]
    selected_mmr_scores = [normalized_scores[first_position]]


    remaining_positions = [position for position in range(len(reranked_indices))
                           if position not in selected_positions]

    while len(selected_positions) < final_k:

        best_position = None
        best_mmr_score = -np.inf

        for position in remaining_positions:

            candidate_index = reranked_indices[position]

            selected_indices = [reranked_indices[selected_position] for selected_position in selected_positions]

            similarities = cosine_similarity([vector_db[candidate_index]], vector_db[selected_indices])[0]

            redundancy = float(np.max(similarities))
            redundancy = np.clip(redundancy, 0.0, 1.0)

            relevance = normalized_scores[position]

            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * redundancy

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_position = position

        selected_positions.append(best_position)
        selected_mmr_scores.append(best_mmr_score)

        remaining_positions.remove(best_position)

    final_indices = reranked_indices[selected_positions]

    return np.array(final_indices), np.array(selected_mmr_scores), normalized_scores
