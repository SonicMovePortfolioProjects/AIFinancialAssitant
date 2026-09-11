import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer

from google.cloud import storage

BUCKET_NAME = "ai-capstone-model-retriever"

def download_from_gcs(src_dir, local_dir):

    print("Connecting to GCS...")

    # Creates a GCS client using the application's
    # Google Cloud credentials
    client = storage.Client()

    # Get bucket
    bucket = client.bucket(BUCKET_NAME)

    # Find all files under models/finbert_relevance_mv/
    blobs = bucket.list_blobs(prefix=src_dir)

    os.makedirs(
        local_dir,
        exist_ok=True
    )

    downloaded = 0

    for blob in blobs:

        # Skip folders
        if blob.name.endswith("/"):
            continue

        relative_path = os.path.relpath(
            blob.name,
            src_dir
        )

        local_path = os.path.join(local_dir, relative_path )

        os.makedirs(
            os.path.dirname(local_path),
            exist_ok=True
        )

        print(
            f"Downloading: gs://{BUCKET_NAME}/{blob.name}"
        )

        blob.download_to_filename(
            local_path
        )

        downloaded += 1

    print(
        f"Downloaded {downloaded} model files"
    )

    return local_dir

#######################################
LOCAL_INDEX = "/tmp/financial_reports.index"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)

# Download FAISS index from GCS

index_blob = bucket.blob(
    "faiss/financial_reports.index"
)

index_blob.download_to_filename(
    LOCAL_INDEX
)

print("FAISS index downloaded")

# Load FAISS index
index = faiss.read_index(
    LOCAL_INDEX
)

print("FAISS index loaded")
print("Number of vectors:", index.ntotal)

# Download metadata from GCS

metadata_blob = bucket.blob(
    "faiss/financial_reports_metadata.pkl"
)

metadata_bytes = metadata_blob.download_as_bytes()

# Load pickle
METADATA = pickle.loads(
    metadata_bytes
)

print("Metadata loaded")
print("Number of metadata records:", len(METADATA))

all_mini_local_dir = download_from_gcs("models/all-MiniLM-L6-v2","/tmp/all-MiniLM-L6-v2")
embedder = SentenceTransformer(
    all_mini_local_dir
)


def retrieve(question):

    # embed question
    query_embedding = embedder.encode(
        [question],
        convert_to_numpy=True  # ,
        # normalize_embeddings=True
    )

    print(query_embedding.shape)
    print(query_embedding[0][:10])

    scores, ids = index.search(
        query_embedding,
        k=50
    )
    print(scores)
    print(ids)
    # return chunks
    records = []
    for score, idx in zip(scores[0], ids[0]):
        chunk = METADATA[idx]
        dict = {}
        dict["score"] = f"{score:.4f}"
        dict["chunk_id"] = chunk["chunk_id"]
        cid = chunk["chunk_id"]
        dict["reference"] = chunk["reference"]
        records.append(dict)
        print(f"{cid}")
    print(f"Number of short listed records: {len(records)}")
    return records