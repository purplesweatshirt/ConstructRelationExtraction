from utils import get_text_tags_and_section_type
from openai import OpenAI
from tqdm import tqdm
import argparse
import signal
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', 
                    default='/path/to/directory/', 
                    help='Base directory')
parser.add_argument('--suffix', 
                    default='_unprocessed_allSections_300_v2.json', 
                    help='Suffix of these files that should be processed')
args = parser.parse_args()


BASE_DIR = args.base_dir
TXT_DIR = os.path.join(BASE_DIR, 'ascii')
SUFFIX   = args.suffix
MODEL = 'gpt-4o-2024-05-13'


with open('key.txt') as key_file:
    KEY = key_file.read()

with open('../prompts/inst_extraction_gpt4.txt') as prompt_file:
    instruct_extraction = prompt_file.read()

with open(os.path.join('../prompts/inst_class.txt'), 'r') as file:
    instruct_classify = file.read()

with open(os.path.join(BASE_DIR, 'filtered_pubmed_v2.json')) as f:
    data = json.load(f)
    xmls = [d['article_id'] + '.xml' for d in data]

client = OpenAI(
    api_key=KEY,
)

existing = os.listdir(os.path.join(BASE_DIR, 'output'))
existing = [f.replace('_const_acr_pipe.json', '.json') for f in existing if '_const_acr_pipe.json' in f]
xmls = [f for f in xmls if not f.replace('.xml', '.json') in existing]
print(len(xmls), "files to be processed!")


for j, xml in enumerate(xmls):

    print(f"Processing file {j+1} of {len(xmls)}: {xml}")
    sents = get_text_tags_and_section_type(os.path.join(BASE_DIR, 'ascii', xml), True)
    
    pred_name = xml.replace('.xml', SUFFIX)

    with open(f'{BASE_DIR}/output/{pred_name}') as f:
        predictions = json.load(f)
        extractions = predictions['extractions']
    
    constructs = []
    for tup in extractions:
        if not str(tup[0]).lower() in constructs and tup[0]:
            constructs.append(tup[0].lower())
        if not str(tup[1]).lower() in constructs and tup[1]:
            constructs.append(tup[1].lower())
        if not str(tup[2]).lower() in constructs and tup[2]:
            constructs.append(tup[2].lower())

    acronyms = []

    for k, c in tqdm(enumerate(constructs)):
        
        rel_sents = [s for s in sents if (f' {c.lower()} ' in s.lower()) or (f' {c.lower()}:' in s.lower()) or (f'({c.lower()})' in s.lower())]
        if len(rel_sents) > 10:
            rel_sents = rel_sents[:10]
        rel_sents_str = '\n'.join(rel_sents)

        messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Only include authentic construct/ latent variable names in your extraction."},
                    {"role": "user", "content": f"Here is a construct (latent variable) name extracted from a scientific article. \n```{c}```\n Decide whether this construct name is an acronym. You should aim for a high recall (identifying all acronyms). Your answer should only consist of yes or no!"},
                    ]

        completion = client.chat.completions.create(model=MODEL,
                                                    messages=messages
                                                    )
        response_str = completion.choices[0].message.content

        if 'yes' in response_str.lower():

            if len(rel_sents) == 0:
                continue

            messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Only include authentic construct/ latent variable names in your extraction."},
                        {"role": "user", "content": f"Here is a construct (latent variable) name extracted from a scientific article which is an acronym. \n```{c}```\n And here is the context from the scientific article containing this name:\n ```{rel_sents_str}```\n\n Create a python dictionary that maps the acronym ({c}) to its true meaning using the context. Your answer should contain no other python code except for this dictionary. Only use knowledge on the acronyms from the context!"},
                        ]

            completion = client.chat.completions.create(model=MODEL,
                                                        messages=messages
                                                        )
            response_str = completion.choices[0].message.content
            acronyms.append(response_str)
            print(response_str)
            print(90*'=')
    

    with open(os.path.join(f'{BASE_DIR}/output', xml.replace('.xml', '_const_acr_pipe.json')), 'w') as f:
        json.dump(acronyms, f)
