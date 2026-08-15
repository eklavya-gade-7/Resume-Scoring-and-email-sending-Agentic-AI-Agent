from sentence_transformers import SentenceTransformer
import numpy as np
import time

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


def create_embeddings(chunks):
    current_time = time.time()

    # encode() takes a list and batches internally - much faster than one at a time
    texts = [chunk["text"] for chunk in chunks]
    vector_db = model.encode(texts)

    print(f'Time taken for embeddings = {time.time() - current_time}')
    return np.array(vector_db)