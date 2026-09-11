from google import genai
import sys
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

import torch

import google.auth
import os
from google.cloud import storage

BUCKET_NAME = "ai-capstone-model-retriever"
MODEL_PREFIX = "models/finbert_relevance_mv"
LOCAL_MODEL_DIR = "/tmp/finbert_relevance_mv"
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

def download_model_from_gcs():

    print("Connecting to GCS...")

    # Creates a GCS client using the application's
    # Google Cloud credentials
    client = storage.Client()

    # Get your bucket
    bucket = client.bucket(BUCKET_NAME)

    # Find all files under models/finbert_relevance_mv/
    blobs = bucket.list_blobs(prefix=MODEL_PREFIX)

    os.makedirs(
        LOCAL_MODEL_DIR,
        exist_ok=True
    )

    downloaded = 0

    for blob in blobs:

        # Skip folders
        if blob.name.endswith("/"):
            continue


        relative_path = os.path.relpath(
            blob.name,
            MODEL_PREFIX
        )

        local_path = os.path.join(
            LOCAL_MODEL_DIR,
            relative_path
        )

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

    return LOCAL_MODEL_DIR


credentials, project = google.auth.default()
print("info")
print(sys.executable)
print(sys.version)
print(torch.__version__)
print(transformers.__version__)
print(torch.__file__)

print("info")


model_path = download_model_from_gcs()
tokenizer = AutoTokenizer.from_pretrained(
    model_path
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)
"""tokenizer = AutoTokenizer.from_pretrained(
    "./model/finbert_relevance_mv"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "./model/finbert_relevance_mv"
)"""

device = torch.device("cpu")

#########

print(torch.__version__)

x = torch.randn(2, 3)
y = torch.randn(3, 4)

print("Before matmul")

z = torch.matmul(x, y)

print("After matmul")
print(z)


print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
#########
#########

def test():
    from transformers import AutoTokenizer, AutoModel
    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    tokenizer = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    model = AutoModel.from_pretrained(
        "distilbert-base-uncased"
    )

    inputs = tokenizer(
        "Hello world",
        return_tensors="pt"
    )

    print("Running model")

    output = model(**inputs)

    print(output)
 ##############

def rerank(question, records):

    torch.set_num_threads(1)
    torch.backends.mkldnn.enabled = False
    reranked = []

    model.eval()

    for chunk in records:
        inputs = tokenizer(
            question,
            chunk["reference"],
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Move tensors to GPU
        try:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except RuntimeError as e:
            print(f"1RuntimeError while scoring chunk: {e}")
            continue
        except Exception as e:
            print(f"1Unexpected error while scoring chunk: {e}")
            continue
        with torch.no_grad():
            try:
                print("Running model on cpu...")
                output = model(**inputs)
                print("Success")
                print(output.logits)
                logits = output.logits

                probability = torch.softmax(
                    logits,
                    dim=1
                    )[0, 1].item()
                reranked.append({
                    "score": probability,
                    "chunk": chunk
                })
            except RuntimeError as e:
                print(f"RuntimeError while scoring chunk: {e}")
                continue

            except Exception as e:
                print(f"Unexpected error while scoring chunk: {e}")
                continue

        reranked.sort(
            key=lambda x: x["score"],
            reverse=True
            )

    top5 = reranked[:5]
    evidence = ""
    total_score = 0
    for i, r in enumerate(top5):
        chunk = r["chunk"]
        total_score += r['score']
        evidence += f"""

                    Evidence {i + 1}
    
                    Chunk ID: {chunk['chunk_id']}

                    Confidence: {r['score']:.3f}

                    {chunk['reference']}

                    ----------------------------------------

                    """
        print(f"{chunk['chunk_id']} -  {r['score']}")
    avg_confidence = total_score/5
    prompt = """Question:
    """f"""{question} Evidence:"""f""" {evidence}""""""Answer using only the evidence above. Cite the section, company, Chunk ID and confidence score of """f"""{avg_confidence:.3f}"""""" back in response"""

    client = genai.Client(
        vertexai=True,
        project="triple-mountain-483601-k3",
        location="global"

    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    if response.text:
        answer = response.text
    else:
        answer = "No answer generated"
    print(answer)
    return answer