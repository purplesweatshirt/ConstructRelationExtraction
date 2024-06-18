from load_model import load_pipeline, load_prompts
from utils import get_text_tags_and_section_type
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
parser.add_argument('--chunk_size', 
                    default=None, 
                    help='Set this to a positive integer, if rather than paragraph the input to the LLM should be a chunk of this size.')
args = parser.parse_args()

CHUNK_SIZE = args.chunk_size
if CHUNK_SIZE:
    CHUNK_SIZE = int(CHUNK_SIZE)
BASE_DIR = args.base_dir
SUFFIX   = args.suffix


pipe = load_pipeline(model_name="microsoft/Phi-3-medium-128k-instruct",
                     tokenizer_name="microsoft/Phi-3-medium-128k-instruct")

generation_args = {"max_new_tokens": 1700,
                   "return_full_text": False,
                   "temperature": 0.0,
                   "do_sample": False,}

instruct_classify, instruct_extraction, _, _, _ = load_prompts(f'{BASE_DIR}/prompts')

with open(os.path.join('../prompts/inst_tables.txt'), 'r') as file:
    instruct_tables = file.read()

with open(os.path.join('../prompts/inst_tables2.txt'), 'r') as file:
    instruct_tables2 = file.read()


def timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, timeout_handler)

with open(f'{BASE_DIR}/filtered_pubmed_v2.json') as f:
    data = json.load(f)[:110]
    xmls = [d['article_id'] + '.xml' for d in data]

existing = os.listdir('output')
existing = [f.replace(SUFFIX, '.json') for f in existing if SUFFIX in f]
xmls = [f for f in xmls if not f.replace('.xml', '.json') in existing]
print(len(xmls), "files to be processed!")


for j, xml in enumerate(xmls):

    print(f"Processing file {j+1} of {len(xmls)}: {xml}")
    sents = get_text_tags_and_section_type(f'{BASE_DIR}/ascii/{xml}')
    chunks = sents

    if CHUNK_SIZE:
        words = []
        chunks = []
        for s in sents:
            if not '[TABLE]' in s:
                words_tmp = s.split()
                for w in words_tmp:
                    words.append(w)

        window_overlap = CHUNK_SIZE // 10
        num_chunks = (len(words) - CHUNK_SIZE) // (CHUNK_SIZE - window_overlap) + 1

        for i in range(num_chunks):
            start = i * (CHUNK_SIZE - window_overlap)
            end = start + CHUNK_SIZE
            chunk = words[start:end]
            chunks.append(' '.join(chunk))
        
    print(len(chunks), "paragraphs to be processed!")
    output = []
    output_unprocessed = []
    output_unprocessed2 = []
    paragraph_ids = []

    for k, s in tqdm(enumerate(chunks)):

        if ("[FIG]" in s) or ("[ABBR]" in s) or (len(s.split()) <= 7):
            continue

        s = s.strip()
        messages = [
                    {"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences."},
                    {"role": "user", "content": f"{instruct_classify} ```{s}```"},
                    ]
        try:
            signal.alarm(240)

            response = pipe(messages, **generation_args)
            response_str = response[0]['generated_text']

            if 'Yes' in response_str:

                if '[TABLE]' in s:
                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research."},
                                {"role": "user", "content": f"{instruct_tables} ```{s}```"},
                                ]

                    response = pipe(messages, **generation_args)
                    response_str = response[0]['generated_text']

                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research."},
                                {"role": "user", "content": f"{instruct_tables2} ```{response_str}```"},
                                ]
                    response = pipe(messages, **generation_args)
                    response_str = response[0]['generated_text']
                    output_unprocessed.append((k, response_str))

                else:
                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences. Only include authentic construct/ latent variable names in your extraction."},
                                {"role": "user", "content": f"{instruct_extraction} {s}\nOutput:"},
                                ]

                    response = pipe(messages, **generation_args)
                    response_str = response[0]['generated_text']
                    output_unprocessed.append((k, response_str))
                    paragraph_ids.append(k)

            signal.alarm(0)

        except TimeoutError:
            print("Skipping sentence due to time limitation of 240s.")

    with open(os.path.join(f'{BASE_DIR}/output', xml.replace('.xml', SUFFIX)), 'w') as f:
        json.dump(output_unprocessed, f)
