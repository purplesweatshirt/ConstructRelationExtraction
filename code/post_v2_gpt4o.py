from utils import extract_dictionaries
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', 
                    default='/path/to/output_directory', 
                    help='Base directory')
parser.add_argument('--suffix', 
                    default='_gpt4o_naive_run2_v1.txt', 
                    help='Suffix for the file')
parser.add_argument('--acr_files', 
                    nargs='+', 
                    help='Suffix for the file',
                    required=True)
args = parser.parse_args()


BASE_DIR = args.base_dir
SUFFIX   = args.suffix
ACR_FILES = args.acr_files


def is_acronym(acronym, phrase):
    words = phrase.split()
    acronym = acronym.lower()
    acronym_length = len(acronym)
    total_length = sum(len(word) for word in words)

    if len(words) >= 10:
        return False
    
    match_count = 0
    acronym_index = 0

    for word in words:
        for char in word:
            if acronym_index < acronym_length and char.lower() == acronym[acronym_index]:
                match_count += 1
                acronym_index += 1

    return match_count > (acronym_length / 2)


def combine_dictionaries(dicts_list):
    combined_dict = {}
    
    for d in dicts_list:
        for key, value in d.items():
            if type(value) == str:
                if is_acronym(key.lower(), value.lower()):
                    combined_dict[key] = value
    
    return combined_dict


files = [f for f in os.listdir(BASE_DIR) if f.endswith(SUFFIX)]

for j, x in enumerate(files):

    print(x)

    with open(os.path.join(BASE_DIR, x)) as file:
        data = json.load(file)
    
    out_name = x.replace(SUFFIX, SUFFIX.replace('_v1', '_v2'))

    acr_names = [x.replace(SUFFIX, acr_name) for acr_name in ACR_FILES]
    
    if not all([os.path.exists(os.path.join(BASE_DIR.replace('_final', ''), acr_name)) for acr_name in acr_names]):
        with open(os.path.join(BASE_DIR, out_name), 'w') as fle:
            json.dump(data, fle)
        continue

    for acr_name in acr_names:
        with open(os.path.join(BASE_DIR.replace('_final', ''), acr_name)) as file:
            acr = json.load(file)
            acr = [extract_dictionaries(a) for a in acr]
            acr = combine_dictionaries(acr)
            acr = {k.lower(): v for k, v in acr.items() if type(k) == str}

        for idx, x in enumerate(data['extractions']):
            if x[0] == '' or x[0] == 'none' or x[0] == 'None':
                data['extractions'][idx][0] = None
            if x[1] == '' or x[1] == 'none' or x[1] == 'None':
                data['extractions'][idx][1] = None
            if x[2] == '' or x[2] == 'none' or x[2] == 'None':
                data['extractions'][idx][2] = None

        for idx, x in enumerate(data['extractions']):
            if x[0] and (x[0].lower() in acr.keys()):
                data['extractions'][idx][0] = acr[x[0].lower()]
            if x[1] and (x[1].lower() in acr.keys()):
                data['extractions'][idx][1] = acr[x[1].lower()]
            if x[2] and (x[2].lower() in acr.keys()):
                data['extractions'][idx][2] = acr[x[2].lower()]

            
    with open(os.path.join(BASE_DIR, out_name), 'w') as fle:
        json.dump(data, fle)
