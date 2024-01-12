# Question 1-  Made by Michael Cedeno
import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to extract text from CSV files
def extract_text_from_csv(file_path, text_column):
    df = pd.read_csv(file_path)
    if text_column in df.columns:
        return df[text_column].tolist()
    else:
        return []

# List of CSV files
csv_files = ["CSV1.csv", "CSV2.csv", "CSV3.csv", "CSV4.csv"]

# Extract text from each CSV file
all_texts = []
for csv_file in csv_files:
    text_list = extract_text_from_csv(csv_file, "TEXT")
    all_texts.extend(text_list)

# Combine all texts into a single string
combined_text = " ".join(all_texts)

# Increase max_length limit
nlp.max_length = len(combined_text) + 1000000

# Process the combined text using spaCy
doc = nlp(combined_text)

# Save the processed text into a .txt file
output_file_path = "combined_text.txt"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(combined_text)

print(f"Text extracted and saved to {output_file_path}")



#Task 2


import pandas as pd
import spacy


# Load spaCy model
nlp = spacy.load("en_core_sci_sm")

# Load scispaCy biomedical NER model
scispacy_linker = ScispaCyEntityLinker()
nlp.add_pipe(scispacy_linker)

# Function to extract text from CSV files
def extract_text_from_csv(file_path, text_column):
    df = pd.read_csv(file_path)
    if text_column in df.columns:
        return df[text_column].tolist()
    else:
        return []

# List of CSV files
csv_files = ["CSV1.csv", "CSV2.csv", "CSV3.csv", "CSV4.csv"]

# Extract text from each CSV file
all_texts = []
for csv_file in csv_files:
    text_list = extract_text_from_csv(csv_file, "TEXT")
    all_texts.extend(text_list)

# Combine all texts into a single string
combined_text = " ".join(all_texts)

# Process the combined text using spaCy
doc = nlp(combined_text)

# Extract biomedical entities using scispaCy
biomedical_entities = []
for ent in doc.ents:
    biomedical_entities.append((ent.text, ent.label_))

# Save the biomedical entities into a .txt file
output_biomed_file_path = "biomedical_entities.txt"
with open(output_biomed_file_path, "w", encoding="utf-8") as output_file:
    for entity, label in biomedical_entities:
        output_file.write(f"{entity}\t{label}\n")

print(f"Biomedical entities extracted and saved to {output_biomed_file_path}")

# Task 3
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# Download NLTK resources (run this once if not already downloaded)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the text from the combined_text.txt file
with open("combined_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text and remove stop words, punctuation, and perform lemmatization
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokens = word_tokenize(text.lower())
filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

# Count the occurrences of each word
word_counts = Counter(filtered_tokens)

# Get the top 30 most common words
top_30_words = word_counts.most_common(30)

# Create a DataFrame for the top 30 words and their counts
df_top_30 = pd.DataFrame(top_30_words, columns=['Word', 'Count'])

# Save the DataFrame to a CSV file
df_top_30.to_csv('top_30_words.csv', index=False)

print("Top 30 words and their counts saved to top_30_words.csv")


# Task 3.2
from collections import Counter
import pandas as pd

# Load the text from the combined_text.txt file
with open("combined_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Choose a pre-trained tokenizer
tokenizer_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Tokenize the text using the AutoTokenizer
tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))

# Count the occurrences of each token
token_counts = Counter(tokens)

# Get the top 30 most common tokens
top_30_tokens = token_counts.most_common(30)

# Create a DataFrame for the top 30 tokens and their counts
df_top_30_tokens = pd.DataFrame(top_30_tokens, columns=['Token', 'Count'])

# Save the DataFrame to a CSV file
df_top_30_tokens.to_csv('top_30_tokens.csv', index=False)

print("Top 30 tokens and their counts saved to top_30_tokens.csv")

# Task 4

import spacy
from spacy import  ScispaCyEntityLinker
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# Load the text from the combined_text.txt file
with open("combined_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# spaCy models
nlp_sci_sm = spacy.load("en_core_sci_sm")
nlp_bc5cdr_md = spacy.load("en_ner_bc5cdr_md")

# BioBERT model
tokenizer_biobert = AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")
model_biobert = AutoModelForTokenClassification.from_pretrained("monologg/biobert_v1.1_pubmed")

# Function to extract entities using spaCy models
def extract_entities_spacy(model, text):
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Function to extract entities using BioBERT
def extract_entities_biobert(model, tokenizer, text):
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="pt")

    outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)

    entities = []
    current_entity = []
    for token, pred in zip(tokens, predictions[0].tolist()):
        if pred == 1:  # BIO scheme: 1 represents 'B' (beginning of an entity)
            current_entity = [token]
        elif pred == 2:  # BIO scheme: 2 represents 'I' (inside an entity)
            current_entity.append(token)
        elif pred == 0 and current_entity:  # BIO scheme: 0 represents 'O' (outside an entity)
            entities.append(" ".join(current_entity))
            current_entity = []

    return entities

# Extract entities using spaCy models
entities_sci_sm = extract_entities_spacy(nlp_sci_sm, text)
entities_bc5cdr_md = extract_entities_spacy(nlp_bc5cdr_md, text)

# Extract entities using BioBERT
entities_biobert = extract_entities_biobert(model_biobert, tokenizer_biobert, text)

# Compare results
common_entities = set(entities_sci_sm) & set(entities_bc5cdr_md) & set(entities_biobert)
difference_sci_sm_bc5cdr_md = set(entities_sci_sm) ^ set(entities_bc5cdr_md)
difference_sci_sm_biobert = set(entities_sci_sm) ^ set(entities_biobert)
difference_bc5cdr_md_biobert = set(entities_bc5cdr_md) ^ set(entities_biobert)

# Print results
print(f"Total entities detected by en_core_sci_sm: {len(entities_sci_sm)}")
print(f"Total entities detected by en_ner_bc5cdr_md: {len(entities_bc5cdr_md)}")
print(f"Total entities detected by BioBERT: {len(entities_biobert)}")
print(f"Common entities: {len(common_entities)}")
print(f"Difference between en_core_sci_sm and en_ner_bc5cdr_md: {len(difference_sci_sm_bc5cdr_md)}")
print(f"Difference between en_core_sci_sm and BioBERT: {len(difference_sci_sm_biobert)}")
print(f"Difference between en_ner_bc5cdr_md and BioBERT: {len(difference_bc5cdr_md_biobert)}")

# Print some example entities for comparison
print("\nExample entities:")
print("en_core_sci_sm:", entities_sci_sm[:5])
print("en_ner_bc5cdr_md:", entities_bc5cdr_md[:5])
print("BioBERT:", entities_biobert[:5])


# Cant install transformser in 3.12 Phyton Version - 6/01 progress task 1







