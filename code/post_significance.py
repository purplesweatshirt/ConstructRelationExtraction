import torch
import argparse
import re, os, json
from tqdm import tqdm
from utils import get_text_tags_and_section_type
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL = 'gpt-4o-2024-05-13'


torch.random.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct",
                                             device_map="cuda",
                                             torch_dtype="auto",
                                             trust_remote_code=True,)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,)

generation_args = {
                    "max_new_tokens": 100,
                    "return_full_text": False,
                    "temperature": 0.0,
                    "do_sample": False,
                  }


def count_numbers_in_string(input_string):
    """
    Counts number of floats in a string, returns an integer
    """
    pattern = re.compile(r'-?\d*\.?\d+')
    matches = pattern.findall(input_string)
    return len(matches)


def extract_number_and_operator(input_string):
    """
    Extract operator and number from a significance string
    Returns tuple (str / None, float / None) of extractions.
    """
    input_string = input_string.replace(" ", "")
    pattern = re.compile(r'([<>]?=?)\s*(-?\d*\.?\d+)|(-?\d*\.?\d+)')

    matches = pattern.search(input_string)

    if matches:
        # if there is an operator present
        if matches.group(1):
            operator = matches.group(1)
            number = float(matches.group(2))
        # if there is no operator, just a number
        else:
            operator = None
            number = float(matches.group(2))
        return operator, number
    return None, None


def categorize(number):
    if number <= 0.001:
        return '< 0.001'
    elif number <= 0.01:
        return '< 0.01'
    elif number <= 0.05:
        return '< 0.05'
    elif number <= 0.1:
        return '< 0.1'
    else:
        return '>= 0.1'



for fname in tqdm(os.listdir(OUTPUT_DIR)):

    if fname.endswith(SUFFIX):

        with open(os.path.join(OUTPUT_DIR, fname)) as in_file:
            data = json.load(in_file)
            extractions = data['extractions']

        path2txt = os.path.join(TXT_DIR, fname.replace(SUFFIX, '.xml'))
        sents = get_text_tags_and_section_type(path2txt, True)

        for idx, tup in enumerate(extractions):

            if tup[-1] and type(tup[-1]) == str:
                if count_numbers_in_string(tup[-1]) == 1:
                    result = extract_number_and_operator(tup[-1])
                    operator, num = result
                    if num >= 1:
                        data['extractions'][idx][-1] = None
                    elif (operator == '=') or (operator == '<') or (not operator):
                        num_str = categorize(num)
                        data['extractions'][idx][-1] = num_str
                    else:
                        data['extractions'][idx][-1] = '>= 0.1'
                elif count_numbers_in_string(tup[-1]) == 0:
                    c_sents = [s for s in sents if (tup[-1] in s) and (not "[FIG]" in s)]
                    c_sents = [s[-250:] for s in c_sents if "[TABLE]" in s]


                    context = '\n'.join(c_sents)

                    if len(context) > 2500:
                        context = context[:2500]

                    messages = [{"role": "system", "content": "You are an expert in stuctural equation modeling research and behavioral sciences."},
                                {"role": "user", "content": f"# Context ```{context}```\n\n# Task\n1. Consider the context above and identify sections that define the meaning of {tup[-1]} in terms of p-values (p < ...). What does {tup[-1]} in this context represent?\n2. Answer with one of the following 'p < 0.001', 'p < 0.01', 'p < 0.05', 'p < 0.1', 'p >= 0.1', if you were able to extract the meaning, or 'None' if no relevant information is contained in the context. Strictly follow these rules."}]
                    response = pipe(messages, **generation_args)
                    response_str = response[0]['generated_text']


                    if '0.001' in response_str:
                        data['extractions'][idx][-1] = '< 0.001'
                    elif '0.01' in response_str:
                        data['extractions'][idx][-1] = '< 0.01'
                    elif '0.05' in response_str:
                        data['extractions'][idx][-1] = '< 0.05'
                    elif '0.1' in response_str and '<' in response_str:
                        data['extractions'][idx][-1] = '< 0.1'
                    elif '0.1' in response_str and '>' in response_str:
                        data['extractions'][idx][-1] = '>= 0.1'
                    else:
                        data['extractions'][idx][-1] = None
                else:
                    data['extractions'][idx][-1] = None
            elif tup[-1] and type(tup[-1]) == float:
                num_str = categorize(tup[-1])
                data['extractions'][idx][-1] = num_str
            else:
                data['extractions'][idx][-1] = None

        out_path = os.path.join(OUTPUT_DIR, fname.replace('_v2.json', '_v3a.json').replace('_v1.json', '_v2a.json'))
        with open(out_path, 'w') as f:
            json.dump(data, f)
