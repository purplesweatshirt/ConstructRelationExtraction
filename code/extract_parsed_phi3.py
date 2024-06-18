from load_model import load_pipeline, load_prompts
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

instruct_classify, instruct_extraction, _, _, _ = load_prompts(f'{BASE_DIR}/prompts')

def timeout_handler(signum, frame):
    raise TimeoutError()


signal.signal(signal.SIGALRM, timeout_handler)

with open(f'{BASE_DIR}/filtered_pubmed_v2.json') as f:
    data = json.load(f)[:110]
    xmls = [d['downloaded_pdf'].replace('.pdf', '.txt') for d in data]

existing = os.listdir('output')
existing = [f.replace(SUFFIX, '.json') for f in existing if SUFFIX in f]
xmls = [f for f in xmls if not f.replace('.xml', '.json') in existing]
print(len(xmls), "files to be processed!")


for j, xml in enumerate(xmls):
    
    if xml == '36687821.xml':
        continue
    
    print(f"Processing file {j+1} of {len(xmls)}: {xml}")
    path2text = os.path.join('parsed_pdf_pymupdf', xml)

    with open(path2text, 'r', encoding='utf-8') as f:
        text = f.read()
        text_words = text.split()

    chunks = []
    window_size    = 100  # number of words
    window_overlap = 20   # number of words
    num_chunks = (len(text_words) - window_size) // (window_size - window_overlap) + 1

    # Create chunks
    for i in range(num_chunks):
        start = i * (window_size - window_overlap)
        end = start + window_size
        chunk = text_words[start:end]
        chunks.append(' '.join(chunk))

    print(len(chunks), "chunks to be processed!")
    
    output_unprocessed = []
    paragraph_ids = []

    for k, s in tqdm(enumerate(chunks)):

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

    outname = data[j]['article_id'] + SUFFIX

    with open(os.path.join(f'{BASE_DIR}/output', outname), 'w') as f:
        json.dump(output_unprocessed, f)
