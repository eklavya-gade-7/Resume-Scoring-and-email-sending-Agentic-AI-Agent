from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from create_chunk_embeddings import model
import numpy as np


TOKEN_PATTERN = r"(?u)C\+\+|C#|[A-Za-z][A-Za-z0-9+#]*(?:[.-][A-Za-z0-9+#]+)*"


def create_sparse_db(chunks):

    chunk_texts = [chunk["text"] for chunk in chunks]

    tfidf = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True)

    try:
        sparse_db = tfidf.fit_transform(chunk_texts)

    except ValueError as error:
        # sklearn reports this as "empty vocabulary; perhaps the documents only
        # contain stop words", which sends you looking in the wrong place -
        # there are no stop words configured here. What it actually means is
        # that TOKEN_PATTERN matched nothing at all, so the resume text is
        # empty or has no words in it. Say that instead.
        raise ValueError(
            f"nothing indexable in {len(chunk_texts)} chunk(s) - the resume "
            f"text is empty or unreadable ({error})"
        ) from error

    return sparse_db, tfidf


def dense_db(subquery, vector_db):

    subquery_embedding = model.encode(subquery)

    similarity_vector = cosine_similarity([subquery_embedding], vector_db)[0]

    return np.array(similarity_vector)


def sparse_retriever(subquery, sparse_db, tfidf):

    subquery_vector = tfidf.transform([subquery])

    similarity_vector = cosine_similarity(subquery_vector, sparse_db)[0]

    return np.array(similarity_vector)


def to_rank_scores(similarity, ignore_zero=False, k=60):

    similarity = np.array(similarity)

    scores = np.zeros(len(similarity))

    if len(similarity) == 0:
        return scores

    if np.max(similarity) == np.min(similarity):
        return scores

    order = np.argsort(similarity)[::-1]

    rank = 1

    for index in order:

        if ignore_zero and similarity[index] <= 0:
            continue

        scores[index] = 1.0 / (k + rank)

        rank += 1

    return scores


def hybrid_retriever(subquery, retrieval_queries, retrieval_type, top_k, vector_db, sparse_db, tfidf):

    retrieval_type = retrieval_type.strip().upper()

    if retrieval_type == "MANUAL":
        return np.array([]), np.array([]), np.array([]), np.array([])

    if retrieval_type not in ["DENSE", "SPARSE", "HYBRID"]:
        raise ValueError(f"Invalid retrieval type: {retrieval_type}")

    if isinstance(retrieval_queries, str):
        retrieval_queries = [query.strip() for query in retrieval_queries.split("||") if query.strip()]

    all_queries = [subquery]

    for query in retrieval_queries:

        query = query.strip()

        if query != "" and query not in all_queries:
            all_queries.append(query)

    dense_rank_vectors = []
    sparse_rank_vectors = []

    dense_raw_vectors = []
    sparse_raw_vectors = []

    for query in all_queries:

        dense_similarity = dense_db(query, vector_db)

        sparse_similarity = sparse_retriever(query, sparse_db, tfidf)

        dense_raw_vectors.append(dense_similarity)

        sparse_raw_vectors.append(sparse_similarity)

        dense_rank_vectors.append(to_rank_scores(dense_similarity))

        sparse_rank_vectors.append(to_rank_scores(sparse_similarity, ignore_zero=True))

    dense_rank_vectors = np.array(dense_rank_vectors)
    sparse_rank_vectors = np.array(sparse_rank_vectors)

    dense_raw_vectors = np.array(dense_raw_vectors)
    sparse_raw_vectors = np.array(sparse_raw_vectors)

    dense_rank_scores = np.max(dense_rank_vectors, axis=0)
    sparse_rank_scores = np.max(sparse_rank_vectors, axis=0)

    dense_raw_scores = np.max(dense_raw_vectors, axis=0)
    sparse_raw_scores = np.max(sparse_raw_vectors, axis=0)

    if retrieval_type == "DENSE":
        dense_weight = 0.7
        sparse_weight = 0.3

    elif retrieval_type == "SPARSE":
        dense_weight = 0.4
        sparse_weight = 0.6

    else:
        dense_weight = 0.5
        sparse_weight = 0.5

    final_similarity = dense_weight * dense_rank_scores + sparse_weight * sparse_rank_scores

    top_k = min(top_k, len(final_similarity))

    top_indices = np.argsort(final_similarity)[::-1][:top_k]

    return top_indices, final_similarity, dense_raw_scores, sparse_raw_scores