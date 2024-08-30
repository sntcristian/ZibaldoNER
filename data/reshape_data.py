import pandas as pd
import spacy
import json
from tqdm import tqdm

# Load the uploaded CSV files
paragraphs_df = pd.read_csv('paragraphs_train.csv')
annotations_df = pd.read_csv('annotations_train.csv')

# Load Spacy model for tokenization
nlp = spacy.load("it_core_news_lg")


# Function to process each document
def process_document(paragraph_id, text, annotations):
    doc = nlp(text)
    tokenized_text = [token.text for token in doc]
    if len(tokenized_text)<=350:
        ner = []
        for _, row in annotations.iterrows():
            start = int(row['start'])
            end = int(row['end'])
            entity_type = row['type']
            if entity_type=="PER":
                entity_type="persona"
            elif entity_type=="LOC":
                entity_type="luogo"
            elif entity_type=="WORK":
                entity_type="opera"
            token_start = None
            token_end = None

            # Finding token boundaries for the annotation
            for i, token in enumerate(doc):
                if token.idx == start:
                    token_start = i
                if token.idx + len(token.text) - 1 == end - 1:
                    token_end = i

            if token_start is not None and token_end is not None:
                ner.append([token_start, token_end, entity_type])
        if len(ner)>0:
            return {
                "id":paragraph_id,
                "tokenized_text": tokenized_text,
                "ner": ner
            }
        else:
            return None
    else:
        return None

pbar = tqdm(total=len(paragraphs_df))

# Merge and process the data
result = []
for _, row in paragraphs_df.iterrows():
    paragraph_id = row['id']
    text = row['text']
    annotations = annotations_df[annotations_df['par_id'] == paragraph_id]
    processed_data = process_document(paragraph_id, text, annotations)
    if processed_data != None:
        result.append(processed_data)
    pbar.update(1)

print(len(result))

# Save the result to a JSON file
output_path = 'json_data/train.json'
with open(output_path, 'w', encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False)
