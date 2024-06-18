from utils import extract_lists
import argparse
import json
import os


parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', 
                    default='/path/to/output_directory', 
                    help='Base directory')
parser.add_argument('--suffix', 
                    default='_gpt4o_naive_run2.txt', 
                    help='Suffix for the file')
parser.add_argument('--naive', 
                    default='False', 
                    help='Sections to exclude for ablation study (choose from [ABSTRACT], [INTRO], [RESULTS], [DISCUSSION], [TABLE])')
args = parser.parse_args()

BASE_DIR = args.base_dir
SUFFIX   = args.suffix
NAIVE  = False if args.naive == 'False' else True


def is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False
    

def filter_unique_tuples(rels):
    relations = [tuple(x) for x in rels]

    seen = {}
    unique_rels = []

    for relation in relations:
        if relation not in seen:
            seen[relation] = 1
            unique_rels.append(relation)
        else:
            seen[relation] += 1

    return [list(x) for x in unique_rels]


files = [f for f in os.listdir(BASE_DIR) if f.endswith(SUFFIX)]
outputs = {}

for j, x in enumerate(files):

    if not 'naive' in x:
        with open(os.path.join(BASE_DIR, x)) as file:
            data = json.load(file)
    else:
        with open(os.path.join(BASE_DIR, x)) as file:
            data = file.read()
    
    output = []
    srcs = []
    output_dict = {}
    print(80*'=')
    print(x)

    if NAIVE:
        loop_range = range(1)
    else:
        loop_range = data

    for item in loop_range:
        
        if NAIVE:
            src_idx = 0
            item = data
        else:
            src_idx = item[0]
            item = item[1]

        relations = extract_lists(item)

        if relations:
            cleaned = []
            if not type(relations[0]) == list:
                continue
            for r in relations:
                if any([type(el) == list for el in r]):
                    s = r
                    for r in s:
                        if len(r) != 5:
                            continue
                        elif (r[0] == None) or (r[2] == None):
                            continue
                        elif (type(r[3]) != float) and (not is_number(r[3])):
                            continue
                        elif (type(r[3]) != float) and (is_number(r[3])):
                            r[3] = float(r[3])
                            cleaned.append(r)
                            srcs.append(src_idx)
                        else:
                            cleaned.append(r)
                            srcs.append(src_idx)
                elif len(r) != 5:
                    continue
                elif (r[0] == None) or (r[2] == None):
                    continue
                elif r[0] == r[2]:
                    continue
                elif (type(r[3]) != float) and (not is_number(r[3])):
                    continue
                elif (type(r[3]) != float) and (is_number(r[3])):
                    r[3] = float(r[3])
                    cleaned.append(r)
                    srcs.append(src_idx)
                else:
                    cleaned.append(r)
                    srcs.append(src_idx)
            output.extend(cleaned)
            
    output = filter_unique_tuples(output)
    output_dict['extractions'] = output
    output_dict['sources'] = srcs
    outputs[x] = output

    outname = SUFFIX.split('.')[0] + '_v1.json'
    with open(os.path.join(BASE_DIR, x.replace(SUFFIX, outname)), 'w') as fle:
        json.dump(output_dict, fle)

    print(output)
