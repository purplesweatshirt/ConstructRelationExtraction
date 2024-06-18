import argparse
import os, json
from itertools import combinations
from utils import get_text_tags_and_section_type
from sentence_transformers import SentenceTransformer


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
embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5')


for fname in os.listdir(OUTPUT_DIR):

    if fname.endswith(SUFFIX):

        with open(os.path.join(OUTPUT_DIR, fname)) as in_file:
            data = json.load(in_file)
            extractions = data['extractions']

        path2txt = os.path.join(TXT_DIR, fname.replace(SUFFIX, '.xml'))
        sents = get_text_tags_and_section_type(path2txt, True)

        constructs = []
        for tup in extractions:
            if not str(tup[0]).lower() in constructs and tup[0]:
                constructs.append(tup[0].lower())
            if not str(tup[1]).lower() in constructs and tup[1]:
                constructs.append(tup[1].lower())
            if not str(tup[2]).lower() in constructs and tup[2]:
                constructs.append(tup[2].lower())

        embeddings = {c: embed_model.encode(c, normalize_embeddings=True) for c in constructs}
        counter = {c: sum([1 for s in sents if c in s]) for c in constructs}

        combs = list(combinations(constructs, 2))

        sim_dict = {}

        for comb in combs:
            embedding1 = embeddings[comb[0]]
            embedding2 = embeddings[comb[1]]

            similarity = embedding1 @ embedding2.T
            norm1, norm2 = embedding1 @ embedding1.T, embedding2 @ embedding2.T
            similarity = similarity / (norm1 * norm2)

            if similarity >= 0.95:
                if counter[comb[0]] >= counter[comb[1]]:
                    key, value = comb[1], comb[0]
                else:
                    key, value = comb[0], comb[1]

                sim_dict[key.lower()] = value.lower()

        for idx, tup in enumerate(extractions):

            if tup[0] and tup[0].lower() in sim_dict.keys():
                data['extractions'][idx][0] = sim_dict[tup[0].lower()]
            if tup[1] and tup[1].lower() in sim_dict.keys():
                data['extractions'][idx][1] = sim_dict[tup[1].lower()]
            if tup[2] and tup[2].lower() in sim_dict.keys():
                data['extractions'][idx][2] = sim_dict[tup[2].lower()]

        out_path = os.path.join(OUTPUT_DIR, fname.replace('.json', '_similarities.json'))
        with open(out_path, 'w') as f:
            json.dump(sim_dict, f)

        out_path = os.path.join(OUTPUT_DIR, fname.replace('_v2.json', '_v3b.json').replace('_v1.json', '_v2b.json'))
        with open(out_path, 'w') as f:
            json.dump(data, f)
