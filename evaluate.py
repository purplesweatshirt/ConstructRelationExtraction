from utils import get_text_tags_and_section_type
import manual_mapping as manual_mapping
import os, json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', 
                    default='', 
                    help='Base directory')
parser.add_argument('--suffix', 
                    default='_gpt4o_pipe_v2.json', 
                    help='Suffix for the file')
parser.add_argument('--exclude_section', 
                    default='', 
                    help='Sections to exclude for ablation study (choose from [ABSTRACT], [INTRO], [METHODS], [RESULTS], [DISCUSS], [TABLE])')
parser.add_argument('--manual', 
                    default='False', 
                    help='whether to include the manual mapping of equivalent construct names in the evaluation.')
args = parser.parse_args()

BASE_DIR = args.base_dir
SUFFIX   = args.suffix
EXCLUDE  = args.exclude_section
MANUAL  = True if args.manual == 'True' else False


with open(f'{BASE_DIR}/filtered_pubmed_final.json', 'r') as f:
    papers = json.load(f)


with open(f'{BASE_DIR}/number_gt.json') as num_file:
    number_of_extractions = json.load(num_file)


no_info = ['art_100.json', 'art_18.json', 'art_19.json', 'art_23.json',
           'art_40.json', 'art_43.json', 'art_46.json', 'art_51.json',
           'art_69.json', 'art_71.json', 'art_83.json', 'art_87.json',
           'art_89.json', 'art_9.json']


sus = []
tps, fns, fps = 0, 0, 0
tps1, fns1, fps1 = 0, 0, 0
tps2, fns2, fps2 = 0, 0, 0
tps3, fns3, fps3 = 0, 0, 0

recalls, precisions,f1_0 = [], [], []
recalls1, precisions1,f1_1 = [], [], []
recalls2, precisions2,f1_2 = [], [], []
recalls3, precisions3, f1_3 = [], [], []
number_gt = []
falses = []
score_per_num = {'<5': [[], [], [], [], [], [], [], [], []],
                 '5-10': [[], [], [], [], [], [], [], [], []],
                 '10-15': [[], [], [], [], [], [], [], [], []],
                 '>15': [[], [], [], [], [], [], [], [], []]}


def apply_mapping(relation, xml):
    
    mapping_dict = manual_mapping.mapping

    if xml in mapping_dict.keys():
        mapping_dict = mapping_dict[xml]
    else:
        return relation

    if relation[0] and relation[0].lower() in mapping_dict.keys():
        relation[0] = mapping_dict[relation[0].lower()]
    if relation[1] and relation[1].lower() in mapping_dict.keys():
        relation[1] = mapping_dict[relation[1].lower()]
    if relation[2] and relation[2].lower() in mapping_dict.keys():
        relation[2] = mapping_dict[relation[2].lower()]
    
    return relation


def print_result_table(recalls, precisions, recalls1, precisions1, recalls2, precisions2, f1_0, f1_1, f1_2, recalls3=None, precisions3=None, f1_3=None):
    print(f"          | Recall | Precision |   F1")
    print(40*'-')
    l1, l2 = len(str(round(sum(recalls)/len(recalls), 3))), len(str(round(sum(precisions)/len(precisions), 3)))
    f1 = 2*(sum(recalls)/len(recalls))*(sum(precisions)/len(precisions))/((sum(recalls)/len(recalls)) + (sum(precisions)/len(precisions)))
    print(f"Entity    | {round(sum(recalls)/len(recalls), 3)}{(7-l1)*' '}| {round(sum(precisions)/len(precisions), 3)}{(10-l2)*' '}| {round(sum(f1_0) / len(f1_0), 3)}")
    l1, l2 = len(str(round(sum(recalls1)/len(recalls1), 3))), len(str(round(sum(precisions1)/len(precisions1), 3)))
    f1 = 2*(sum(recalls1)/len(recalls1))*(sum(precisions1)/len(precisions1))/((sum(recalls1)/len(recalls1)) + (sum(precisions1)/len(precisions1)))
    print(f"3-Tuple   | {round(sum(recalls1)/len(recalls1), 3)}{(7-l1)*' '}| {round(sum(precisions1)/len(precisions1), 3)}{(10-l2)*' '}| {round(sum(f1_1) / len(f1_1), 3)} ")
    l1, l2 = len(str(round(sum(recalls2)/len(recalls2), 3))), len(str(round(sum(precisions2)/len(precisions2), 3)))
    f1 = 2*(sum(recalls2)/len(recalls2))*(sum(precisions2)/len(precisions2))/((sum(recalls2)/len(recalls2)) + (sum(precisions2)/len(precisions2)))
    print(f"4-Tuple   | {round(sum(recalls2)/len(recalls2), 3)}{(7-l1)*' '}| {round(sum(precisions2)/len(precisions2), 3)}{(10-l2)*' '}| {round(sum(f1_2) / len(f1_2), 3)} ")
    if (recalls3 and precisions3 and f1_3):
        l1, l2 = len(str(round(sum(recalls3)/len(recalls3), 3))), len(str(round(sum(precisions3)/len(precisions3), 3)))
        f1 = 2*(sum(recalls3)/len(recalls3))*(sum(precisions3)/len(precisions3))/((sum(recalls3)/len(recalls3)) + (sum(precisions3)/len(precisions3)))
        print(f"5-Tuple   | {round(sum(recalls3)/len(recalls3), 3)}{(7-l1)*' '}| {round(sum(precisions3)/len(precisions3), 3)}{(10-l2)*' '}| {round(sum(f1_3) / len(f1_3), 3)} ")
    

def f1(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2*(precision*recall)/(precision+recall)


def extract_tuples(extraction_data):
    return [(rel.get("Construct from", "").lower(), rel.get("Construct Moderator", "").lower(), rel.get("Construct to", "").lower())
            for rel in extraction_data['relations'] if rel.get("Construct from") and rel.get("Construct to")]


def extract_tuples(extraction_data, include_pc=False, include_significance=False):
    if include_pc:
        if not include_significance:
            return [(rel.get("Construct from", "").lower(), rel.get("Construct Moderator", "").lower(), rel.get("Construct to", "").lower(), str(rel.get("Path coefficient", "")))
                    for rel in extraction_data['relations'] if rel.get("Construct from") and rel.get("Construct to")]
        else:
            return [(rel.get("Construct from", "").lower(), rel.get("Construct Moderator", "").lower(), rel.get("Construct to", "").lower(), str(rel.get("Path coefficient", "")), str(rel["significance"]).replace(' ', '').strip().lower())
                    for rel in extraction_data['relations'] if rel.get("Construct from") and rel.get("Construct to")]

    else:
        return [(rel.get("Construct from", "").lower(), rel.get("Construct Moderator", "").lower(), rel.get("Construct to", "").lower())
            for rel in extraction_data['relations'] if rel.get("Construct from") and rel.get("Construct to")]


def extract_tuples_preds(extraction_data, include_pc=False, include_significance=False):
    tuples = []
    for rel in extraction_data:
        const_from, const_mod, const_to = rel[0], rel[1], rel[2]
        if const_mod and not const_to:
            const_to, const_mod = const_mod, const_to
        if str(const_from).isdigit():
            const_from = None
        if str(const_to).isdigit():
            const_to = None
        if str(const_mod).isdigit():
            const_mod = None
        if include_pc and not include_significance:
            tuples.append((str(const_from).lower(), str(const_mod).lower(), str(const_to).lower(), str(rel[3])))
        elif include_pc and include_significance:
            tuples.append((str(const_from).lower(), str(const_mod).lower(), str(const_to).lower(), str(rel[3]), str(rel[4]).replace(' ', '').strip().lower()))
        else:
            tuples.append((str(const_from).lower(), str(const_mod).lower(), str(const_to).lower()))
    return tuples

number_of_extractions = {}


print(len(os.listdir(f'{BASE_DIR}/Test_LabelsPubmed')))
for indx, fname in enumerate(sorted(os.listdir(f'{BASE_DIR}/Test_LabelsPubmed'))):

    with open(f'{BASE_DIR}/Test_LabelsPubmed/{fname}', 'r') as f:
        extractions = json.load(f)

    doi_url = extractions['source']
    number_of_extractions[fname] = len(extractions['relations'])

    paper = None

    for p in papers:
        if p['doi_url'] == doi_url:
            paper = p
            break
    
    constructs = []
    for gt in extractions['relations']:
        if gt["Construct from"] and gt["Construct from"] not in constructs:
            constructs.append(gt["Construct from"])
        if gt["Construct to"] and gt["Construct to"] not in constructs:
            constructs.append(gt["Construct to"])
        if gt["Construct Moderator"] and gt["Construct Moderator"] not in constructs:
            constructs.append(gt["Construct Moderator"])

    constructs = list(set(constructs))
    constructs = [c.lower() for c in constructs if c and c != 'None']

    if paper == None:
        print(fname.upper())
        continue

    #if str(paper["article_id"])== '36687821':
    #    continue

    sents = get_text_tags_and_section_type(os.path.join(BASE_DIR, 'ascii', f'{paper["article_id"]}.xml'))

    if os.path.exists(f'{BASE_DIR}/output_final/{paper["article_id"]}{SUFFIX}'):
        with open(f'{BASE_DIR}/output_final/{paper["article_id"]}{SUFFIX}', 'r') as f:
            preds = json.load(f)

        # Ignore specific section (EXCLUDE)
        if EXCLUDE:
            preds_inds = [k for k, _ in enumerate(preds['extractions']) if not (EXCLUDE in sents[preds['sources'][k]])]
            preds['extractions'] = [preds['extractions'][k] for k in preds_inds]
            preds['sources'] = [preds['sources'][k] for k in preds_inds]

        if MANUAL:
            preds['extractions'] = [apply_mapping(pred, paper["article_id"]+'.xml') for pred in preds['extractions']]
        
        preds = preds['extractions']
        preds = [p for p in preds if ((not ((len(p[0]) <= 4) or (len(str(p[2])) <= 4))) and (str(p[0]).lower() != str(p[2]).lower()))]

        constructs_preds = []
        for p in preds:
            if p[0] and p[0] not in constructs_preds:
                constructs_preds.append(p[0])
            if p[1] and p[1] not in constructs_preds:
                constructs_preds.append(p[1])
            if p[2] and p[2] not in constructs_preds:
                constructs_preds.append(p[2])

        constructs_preds = list(set(constructs_preds))
        constructs_preds = [str(c).lower() for c in constructs_preds if c and c != 'None']
    else:
        continue

    precision, recall = 0, 0
    if len(constructs_preds) > 0:
        precision = round(len(set(constructs).intersection(set(constructs_preds))) / len(constructs_preds), 3)
    if len(constructs) > 0:
        recall = round(len(set(constructs_preds).intersection(set(constructs))) / len(constructs), 3)

    tps += len(set(constructs_preds).intersection(set(constructs)))
    fns += len(set(constructs).difference(set(constructs_preds)))
    fps += len(set(constructs_preds).difference(set(constructs)))

              
    if recall == 0:
        sus.append(paper)

    recalls.append(recall)
    precisions.append(precision)
    f1_0.append(f1(precision, recall))
    
    if number_of_extractions[fname] < 5:
        score_per_num['<5'][0].append(recall)
        score_per_num['<5'][1].append(precision)
        score_per_num['<5'][6].append(f1(precision, recall))
    elif number_of_extractions[fname] < 10:
        score_per_num['5-10'][0].append(recall)
        score_per_num['5-10'][1].append(precision)
        score_per_num['5-10'][6].append(f1(precision, recall))
    elif number_of_extractions[fname] < 15:
        score_per_num['10-15'][0].append(recall)
        score_per_num['10-15'][1].append(precision)
        score_per_num['10-15'][6].append(f1(precision, recall))
    else:
        score_per_num['>15'][0].append(recall)
        score_per_num['>15'][1].append(precision)
        score_per_num['>15'][6].append(f1(precision, recall))
    
    # evaluate tuples of construct from, construct to, construct moderator
    gt_tuples = extract_tuples(extractions)
    pred_tuples = extract_tuples_preds(preds)

    if pred_tuples:
        precision = round(len(set(pred_tuples).intersection(set(gt_tuples))) / len(pred_tuples), 3)
    if gt_tuples:
        recall = round(len(set(gt_tuples).intersection(set(pred_tuples))) / len(gt_tuples), 3)

    tps1 += len(set(pred_tuples).intersection(set(gt_tuples)))
    fns1 += len(set(gt_tuples).difference(set(pred_tuples)))
    fps1 += len(set(pred_tuples).difference(set(gt_tuples)))

    recalls1.append(recall)
    precisions1.append(precision)
    f1_1.append(f1(precision, recall))
    
    if number_of_extractions[fname] < 5:
        score_per_num['<5'][2].append(recall)
        score_per_num['<5'][3].append(precision)
        score_per_num['<5'][7].append(f1(precision, recall))
    elif number_of_extractions[fname] < 10:
        score_per_num['5-10'][2].append(recall)
        score_per_num['5-10'][3].append(precision)
        score_per_num['5-10'][7].append(f1(precision, recall))
    elif number_of_extractions[fname] < 15:
        score_per_num['10-15'][2].append(recall)
        score_per_num['10-15'][3].append(precision)
        score_per_num['10-15'][7].append(f1(precision, recall))
    else:
        score_per_num['>15'][2].append(recall)
        score_per_num['>15'][3].append(precision)
        score_per_num['>15'][7].append(f1(precision, recall))
    
    gt_tuples = extract_tuples(extractions, include_pc=True)
    number_gt.append(len(gt_tuples))
    pred_tuples = extract_tuples_preds(preds, include_pc=True)

    if pred_tuples:
        precision = round(len(set(pred_tuples).intersection(set(gt_tuples))) / len(pred_tuples), 3)
    if gt_tuples:
        recall = round(len(set(gt_tuples).intersection(set(pred_tuples))) / len(gt_tuples), 3)

    tps2 += len(set(pred_tuples).intersection(set(gt_tuples)))
    fns2 += len(set(gt_tuples).difference(set(pred_tuples)))
    fps2 += len(set(pred_tuples).difference(set(gt_tuples)))

    recalls2.append(recall)
    precisions2.append(precision)
    f1_2.append(f1(precision, recall))
    
    if number_of_extractions[fname] < 5:
        score_per_num['<5'][4].append(recall)
        score_per_num['<5'][5].append(precision)
        score_per_num['<5'][8].append(f1(precision, recall))
    elif number_of_extractions[fname] < 10:
        score_per_num['5-10'][4].append(recall)
        score_per_num['5-10'][5].append(precision)
        score_per_num['5-10'][8].append(f1(precision, recall))
    elif number_of_extractions[fname] < 15:
        score_per_num['10-15'][4].append(recall)
        score_per_num['10-15'][5].append(precision)
        score_per_num['10-15'][8].append(f1(precision, recall))
    else:
        score_per_num['>15'][4].append(recall)
        score_per_num['>15'][5].append(precision)
        score_per_num['>15'][8].append(f1(precision, recall))


    gt_tuples = extract_tuples(extractions, include_pc=True, include_significance=True)
    number_gt.append(len(gt_tuples))
    pred_tuples = extract_tuples_preds(preds, include_pc=True, include_significance=True)

    if pred_tuples:
        precision = round(len(set(pred_tuples).intersection(set(gt_tuples))) / len(pred_tuples), 3)
    if gt_tuples:
        recall = round(len(set(gt_tuples).intersection(set(pred_tuples))) / len(gt_tuples), 3)

    tps3 += len(set(pred_tuples).intersection(set(gt_tuples)))
    fns3 += len(set(gt_tuples).difference(set(pred_tuples)))
    fps3 += len(set(pred_tuples).difference(set(gt_tuples)))

    recalls3.append(recall)
    precisions3.append(precision)
    f1_3.append(f1(precision, recall))


print(len(recalls))
print_result_table(recalls, precisions, recalls1, precisions1, recalls2, precisions2, f1_0, f1_1, f1_2, recalls3, precisions3, f1_3)
print(50*'=')
for k, v in score_per_num.items():
    print(k, "construct relations in the paper")
    print()
    print_result_table(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
    print()


# Uncomment for evaluation scores on construct relation level
'''
r, p = round(tps/(tps+fns), 3), round(tps/(tps+fps), 3)
f1 = 2*r*p/(r+p)
print("Recall relation level (entity):", round(tps/(tps+fns), 3))
print("Precision relation level (entity):", round(tps/(tps+fps), 3))
print("F1 relation level (entity):", round(f1, 3))    
r, p = round(tps1/(tps1+fns1), 3), round(tps1/(tps1+fps1), 3)
f1 = 2*r*p/(r+p)
print("Recall relation level (3-tuple):", r)
print("Precision relation level (3-tuple):", p)
print("F1 relation level (3-tuple):", round(f1, 3))
r, p = round(tps2/(tps2+fns2), 3), round(tps2/(tps2+fps2), 3)
f1 = 2*r*p/(r+p)
print("Recall relation level (4-tuple):", r)
print("Precision relation level (4-tuple):", p)
print("F1 relation level (4-tuple):", round(f1, 3)) 
r, p = round(tps3/(tps3+fns3), 3), round(tps3/(tps3+fps3), 3)
f1 = 2*r*p/(r+p)
print("Recall relation level (5-tuple):", r)
print("Precision relation level (5-tuple):", p)
print("F1 relation level (5-tuple):", round(f1, 3))    
'''
