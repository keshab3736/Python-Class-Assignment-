import os
import zipfile
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk


nltk.download('punkt')


zip_folder_path ='/Users/aashirkandel/Desktop/assigment1/Assigment.csv.zip'
extract_folder_path = '/Users/aashirkandel/Desktop/python/extrated_folder'
with zipfile.ZipFile(zip_folder_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder_path)

csv_files = [file for file in os.listdir(extract_folder_path) if file.endswith('.csv')]


all_text = []

for csv_file in csv_files:
    file_path = os.path.join(extract_folder_path, csv_file)

    df = pd.read_csv(file_path)

    large_text_column = 'Name'
   
    all_text.extend(df[large_text_column].dropna().tolist())


merged_text = ' '.join(all_text)


sentences = sent_tokenize(merged_text)


output_txt_file = 'output_text.txt'
with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
    for sentence in sentences:
        txt_file.write(sentence + '\n')

print(f'Text extracted and saved to {output_txt_file}')
