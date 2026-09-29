# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
import spacy

print("Downloading sentence transformer...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Done. Model cached at:", model.cache_folder)

print("Checking spacy...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("Spacy model loaded.")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
