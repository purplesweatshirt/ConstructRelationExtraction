import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)


def load_pipeline(model_name="microsoft/Phi-3-mini-128k-instruct",
                  tokenizer_name="microsoft/Phi-3-mini-128k-instruct"):

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="cuda",
                                                torch_dtype="auto",
                                                trust_remote_code=True,)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,)
    
    return pipe


def load_prompts(path2prompts):

    with open(os.path.join(path2prompts, 'inst_extraction.txt'), 'r') as file:
        instruct_extraction = file.read()

    with open(os.path.join(path2prompts, 'inst_tables.txt'), 'r') as file:
        instruct_tables = file.read()
    
    with open(os.path.join(path2prompts, 'inst_tables2.txt'), 'r') as file:
        instruct_tables2 = file.read()

    with open(os.path.join(path2prompts, 'inst_class.txt'), 'r') as file:
        instruct_classify = file.read()

    with open(os.path.join(path2prompts, 'inst_class_tables.txt'), 'r') as file:
        instruct_classify_tables = file.read()

    return instruct_classify, instruct_extraction, instruct_classify_tables, instruct_tables, instruct_tables2

