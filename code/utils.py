import xml.etree.ElementTree as ET
import json
import ast
import os
import re


def extract_lists(input_string):
    cleaned_string = input_string.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    # The pattern looks only for simple lists.
    pattern = r'\[([^\[\]]*)\]'
    try:
        matches = re.findall(pattern, cleaned_string, re.DOTALL)
        simple_lists = []
        for m in matches:
            # Wrapping the matched content with square brackets to evaluate it as a list.
            try:
                parsed_list = ast.literal_eval(f'[{m}]')
            except Exception as e:
                continue
            if isinstance(parsed_list, list):
                simple_lists.extend([parsed_list])
        return simple_lists
    except Exception as e:
        return []


def extract_dictionaries(input_string):
    cleaned_string = input_string.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')

    # The pattern looks for dictionaries.
    pattern = r'\{[^\{\}]*\}'
    try:
        matches = re.findall(pattern, cleaned_string, re.DOTALL)
        combined_dict = {}
        for m in matches:
            # Evaluating the matched content as a dictionary.
            parsed_dict = ast.literal_eval(m)
            if isinstance(parsed_dict, dict):
                combined_dict.update(parsed_dict)
        return combined_dict
    except Exception as e:
        return {}

    

def get_text_tags_and_section_type(file_path, sentence_level=False):
    tree = ET.parse(file_path)
    root = tree.getroot()

    passages = root.findall(".//passage")

    prev_section_type = None
    table_text = ""
    paragraphs = []

    for passage in passages:
        text_tag = passage.find("text")
        infon_tag = passage.find("infon[@key='section_type']")
        infon_key_tag = passage.find("infon[@key='type']")

        if text_tag is not None:
            text = text_tag.text
        else:
            text = None

        if infon_tag is not None:
            section_type = infon_tag.text
        else:
            section_type = None

        if infon_key_tag is not None:
            section_key = infon_key_tag.text
        else:
            section_key = None

        if section_type == 'TABLE':
            new_table = ((prev_key_type == 'table_footnote') and (section_key == 'table_caption')) or ((prev_key_type == 'table') and (section_key == 'table_caption')) or ((prev_key_type != 'table_caption') and (section_key == 'table'))

            # If and old table should be continued
            if (prev_section_type == 'TABLE') and not new_table:
                table_text += " " + text
            # If we should start a new table
            else:
                if table_text:
                    paragraphs.append("[TABLE] " + table_text)
                table_text = text
        else:
            if table_text:
                paragraphs.append("[TABLE] " + table_text)
                table_text = ""

            if not section_type in ['REF', 'AUTH_CONT', 'ACK_FUND', 'COMP_INT', 'SUPPL']:

                if not sentence_level:
                    paragraphs.append(f"[{section_type}] " + text)
                elif sentence_level:
                    # Split text into sentences and append the sentences
                    sentences = text.split('. ')
                    for sentence in sentences:
                        paragraphs.append(f"[{section_type}] " + sentence)

        prev_section_type = section_type
        prev_key_type = section_key

    if table_text:
        paragraphs.append("[TABLE] " + table_text)

    return paragraphs
