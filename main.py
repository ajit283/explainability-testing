import jsonlines
import cohere
import pickle
import os
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture
import numpy as np

load_dotenv()
# Initialize Cohere client
co = cohere.Client(os.getenv('COHERE_API_KEY'))

def read_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def generate_embeddings(texts):
    response = co.embed(texts=texts, model='large')
    return response.embeddings

def save_embeddings(file_path, embeddings):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def get_embeddings(jsonl_path, pkl_path):
    if os.path.exists(pkl_path):
        print(f"Loading embeddings from {pkl_path}")
        return load_embeddings(pkl_path)
    else:
        print(f"{pkl_path} not found. Generating new embeddings.")
        data = read_jsonl(jsonl_path)
        texts = [item['text'] for item in data]
        embeddings = generate_embeddings(texts)
        save_embeddings(pkl_path, embeddings)
        return embeddings


def main():
    jsonl_path = 'nfcorpus/corpus.jsonl'
    pkl_path = 'embeddings.pkl'
    
    # Step 1: Get embeddings (load from file or generate)
    embeddings = get_embeddings(jsonl_path, pkl_path)
    
    # Optionally, you can do something with the embeddings here
    print(f"Got {len(embeddings)} embeddings.")

    # convert embeddings to np array
    embeddings_np = np.array(embeddings)   


    gmm = GaussianMixture(n_components=5) 
    
    # Fit the GMM to the embeddings
    gmm.fit(embeddings)

    medical_string = """
        BACKGROUND: Preclinical studies have shown that statins, particularly simvastatin, can prevent growth in breast cancer cell lines and animal models
        """

    test_strings = ["I am a test", "This is something medical", medical_string]

    test_embeddings = np.array(generate_embeddings(test_strings))

    log_probs = gmm.score_samples(test_embeddings)

    print(f"Log probability of test embeddings: {log_probs}")

    normal_probs = np.exp(log_probs)

    print(f"Normal probability of test embeddings: {normal_probs}")



if __name__ == "__main__":
    main()

