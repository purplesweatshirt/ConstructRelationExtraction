from load_model import load_pipeline, load_prompts
from utils import get_text_tags_and_section_type, extract_lists
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
                    default='_gpt4o_pipe_v2.json', 
                    help='Suffix for the file')
args = parser.parse_args()

BASE_DIR = args.base_dir
SUFFIX   = args.suffix

pipe = load_pipeline(model_name="microsoft/Phi-3-medium-128k-instruct",
                     tokenizer_name="microsoft/Phi-3-medium-128k-instruct")

generation_args = {"max_new_tokens": 1700,
                   "return_full_text": False,
                   "temperature": 0.0,
                   "do_sample": False,}

with open(f'{BASE_DIR}/filtered_pubmed_v2.json') as f:
    data = json.load(f)[:110]
    xmls = [d['article_id'] + '.xml' for d in data]

existing = os.listdir('output')
existing = [f.replace('_const_acr_table.json', '.json') for f in existing if ('_const_acr_parsed.json' in f) and (f.replace('_const_acr_parsed.json', '_const_map_parsed.json') in existing)]
xmls = [f for f in xmls if not f.replace('.xml', '.json') in existing]
print(len(xmls), "files to be processed!")


for j, xml in enumerate(xmls):
    
    if xml == '36687821.xml':
        continue
    
    print(f"Processing file {j+1} of {len(xmls)}: {xml}")
    sents = get_text_tags_and_section_type(f'{BASE_DIR}/ascii/{xml}', True)
    
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

    # Same constructs, slightly different names
    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences."},
                {"role": "user", "content": f"Here is a list (indicated by three tickmarks) of construct (latent variable) names extracted from a scientific article. \n```{constructs}```\n Check this list for constructs that are equivalent, i.e., refer to the exact same construct. They truly have to mean the same thing. Your output should be a python dictionary which maps constructs to their equivalent name from the list if there is one. This means both keys and values have to be string values of a single construct name. Your answer should contain no other python code except for this dictionary."},
                ]

    response = pipe(messages, **generation_args)
    response_str = response[0]['generated_text']
    print(response_str)

    with open(os.path.join(f'{BASE_DIR}/output', xml.replace('.xml', '_const_map_parsed.json')), 'w') as f:
        json.dump(response_str, f)
    print(90*'=')

    # Acronyms
    for k, c in tqdm(enumerate(constructs)):
        
        rel_sents = [s for s in sents if (f' {c.lower()} ' in s.lower()) or (f' {c.lower()}:' in s.lower()) or (f'({c.lower()})' in s.lower())]
        if len(rel_sents) > 10:
            rel_sents = rel_sents[:10]
        rel_sents_str = '\n'.join(rel_sents)

        messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Answer only with yes or no!"},
                    {"role": "user", "content": f"Here is a construct (latent variable) name extracted from a scientific article. \n```{c}```\n Decide whether this construct name is an acronym. You should aim for a high recall (identifying all acronyms). Your answer should only consist of yes or no!"},
                    ]
        response = pipe(messages, **generation_args)
        response_str = response[0]['generated_text']

        if 'yes' in response_str.lower():

            if len(rel_sents) == 0:
                continue

            messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Answer only with yes or no!"},
                        {"role": "user", "content": f"Here is a construct (latent variable) name extracted from a scientific article which is an acronym. \n```{c}```\n And here is the context from the scientific article containing this name:\n ```{rel_sents_str}```\n\n Create a python dictionary that maps the acronym ({c}) to its true meaning using the context. Your answer should contain no other python code except for this dictionary. Only use knowledge on the acronyms from the context!"},
                        ]
            response = pipe(messages, **generation_args)
            response_str = response[0]['generated_text']
            messages.append({"role": "assistant", "content": response_str})
            messages.append({"role": "user", "content": "Carefully revise your answer. Does the mapping of the acronym truly reflect its meaning in this context? Critically reflect and output a revised mapping dictionary (python)."})
            response = pipe(messages, **generation_args)
            response_str = response[0]['generated_text']
            acronyms.append(response_str)
            print(response_str)
            print(90*'=')
    
    with open(os.path.join(f'{BASE_DIR}/output', xml.replace('.xml', '_const_acr_parsed.json')), 'w') as f:
        json.dump(acronyms, f)
    
