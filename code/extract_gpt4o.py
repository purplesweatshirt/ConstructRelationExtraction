from utils import get_text_tags_and_section_type
from openai import OpenAI
from tqdm import tqdm
import signal
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', 
                    default='/path/to/directory/', 
                    help='Base directory')
parser.add_argument('--suffix', 
                    default='_gpt4o_pipe_unprocessed.json', 
                    help='Suffix for the file')
args = parser.parse_args()


BASE_DIR = args.base_dir
SUFFIX   = args.suffix
MODEL = 'gpt-4o-2024-05-13'


def timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, timeout_handler)


with open('key.txt') as key_file:
    KEY = key_file.read()

with open('../prompts/inst_extraction_gpt4.txt') as prompt_file:
    instruct_extraction = prompt_file.read()

with open(os.path.join('../prompts/inst_class.txt'), 'r') as file:
    instruct_classify = file.read()

with open(os.path.join('../prompts/inst_tables.txt'), 'r') as file:
    instruct_tables = file.read()

with open(os.path.join('../prompts/inst_tables2.txt'), 'r') as file:
    instruct_tables2 = file.read()

with open(os.path.join(BASE_DIR, 'filtered_pubmed_v2.json')) as f:
    data = json.load(f)
    xmls = [d['article_id'] + '.xml' for d in data]

client = OpenAI(
    api_key=KEY,
)

existing = os.listdir(os.path.join(BASE_DIR, 'output'))
existing = [f.replace(SUFFIX, '.json') for f in existing if SUFFIX in f]
xmls = [f for f in xmls if not f.replace('.xml', '.json') in existing]
print(len(xmls), "files to be processed!")


for j, xml in enumerate(xmls):

    if xml == '36687821.xml':
        continue

    print(f"Processing file {j+1} of {len(xmls)}: {xml}")
    sents = get_text_tags_and_section_type(os.path.join(BASE_DIR, 'ascii', xml))
    print(len(sents), "paragraphs to be processed!")
    output = []
    output_unprocessed = []
    output_unprocessed2 = []
    paragraph_ids = []

    for k, s in tqdm(enumerate(sents)):

        if ("[TITLE]" in s) or ("[FIG]" in s) or ("[ABBR]" in s) or (len(s.split()) <= 7):
            continue

        s = s.strip()
        messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences."},
                    {"role": "user", "content": f"{instruct_classify} ```{s}```"},
                    ]
        try:
            signal.alarm(180)

            completion = client.chat.completions.create(model=MODEL,
                                                        messages=messages
                                                        )
            response_str = completion.choices[0].message.content

            if 'Yes' in response_str:
                
                if '[TABLE]' in s:
                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research."},
                                {"role": "user", "content": f"{instruct_tables} ```{s}```"},
                                ]

                    completion = client.chat.completions.create(model=MODEL,
                                                                messages=messages
                                                                )
                    response_str = completion.choices[0].message.content

                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research."},
                                {"role": "user", "content": f"{instruct_tables2} ```{response_str}```"},
                                ]
                    
                    completion = client.chat.completions.create(model=MODEL,
                                                                messages=messages
                                                                )
                    response_str = completion.choices[0].message.content
                    output_unprocessed.append((k, response_str))
                else:
                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Only include authentic construct/ latent variable names in your extraction."},
                                {"role": "user", "content": f"{instruct_extraction} {s}\nOutput:"},
                                ]

                    completion = client.chat.completions.create(model=MODEL,
                                                                messages=messages
                                                                )
                    response_str = completion.choices[0].message.content
                    output_unprocessed.append((k, response_str))

            signal.alarm(0)

        except TimeoutError:
            print("Skipping sentence due to time limitation of 180s.")

    with open(os.path.join(BASE_DIR, 'output', xml.replace('.xml', SUFFIX)), 'w') as f:
        json.dump(output_unprocessed, f)
