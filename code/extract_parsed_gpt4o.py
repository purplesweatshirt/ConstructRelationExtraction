from openai import OpenAI
from tqdm import tqdm
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', 
                    default='/path/to/directory/', 
                    help='Base directory')
parser.add_argument('--suffix', 
                    default='_gpt4o_naive_parsed.json', 
                    help='Suffix for the file')
args = parser.parse_args()


BASE_DIR = args.base_dir
SUFFIX   = args.suffix
MODEL = 'gpt-4o-2024-05-13'


with open('key.txt') as key_file:
    KEY = key_file.read()

with open('../prompts/inst_extraction_gpt4.txt') as prompt_file:
    instruct_extract = prompt_file.read()


with open(os.path.join(BASE_DIR, 'filtered_pubmed_v2.json')) as f:
    data = json.load(f)
    xmls = [d['downloaded_pdf'].replace('.pdf', '.txt') for d in data]
    output_files = os.listdir(os.path.join(BASE_DIR, 'output'))
    existing = [f['downloaded_pdf'].replace('.pdf', '.txt') for f in data if (f['article_id'] + SUFFIX) in output_files]
    xmls2 = [xml for xml in xmls if not xml in existing]

client = OpenAI(
    api_key=KEY,
)

print(len(xmls2), "files to process!")


for j, xml in tqdm(enumerate(xmls2)):

    path2text = os.path.join('../parsed_pdf_pymupdf', xml)

    with open(path2text, 'r', encoding='utf-8') as f:
        input_text = f.read()
    
    completion = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Only include authentic construct/ latent variable names in your extraction."},
            {"role": "user", "content": f"{instruct_extract} ```{input_text}```"}
            ]
    )

    response_msg = completion.choices[0].message.content
    outname = data[j]['article_id'] + SUFFIX

    with open(os.path.join(BASE_DIR, 'output', outname), 'w') as output_file:
        output_file.write(response_msg)
