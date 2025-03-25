""" Preprocess dataset for knights and knaves logic task """

import os
from datasets import Dataset, load_dataset, concatenate_datasets
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import json

def make_prefix(dp, template_type):
    quiz = dp['quiz']
    if template_type == 'base':
        prefix = f"""The user asks a question, and the Assistant solves it.The assistant first thinks about the reasoning process in the mind and then provides the user with the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. List the identity of each person one by one, for example, <answer> (1) Zoey is a knight\n(2) Oliver is a knight\n(3)... </answer>.\n\nUser:{quiz}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.\n<|im_end|>\n<|im_start|>user\n{quiz}\n<|im_end|>\n<|im_start|>assistant\n<think>"""
    return prefix

system_prompt = f'''You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a logical reasoning problem. After thinking, when you finally reach a conclusion, clearly state the identity of each character within <answer> </answer> tags. i.e., <answer> (1) Zoey is a knight\n(2) ... </answer>.'''



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/curriculum-llm/data/kk-difficulty')
    # parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    data_source = 'K-and-K/knights-and-knaves'

    train_dataset = load_dataset(data_source, "train")
    test_dataset = load_dataset(data_source, 'test')

    def merge_kk_dataset(dataset):
        processed_datasets = []
        for split_name in dataset.keys():
            n_people = int(split_name.replace('ppl', ''))
            ds = dataset[split_name].map(lambda example: {'n_ppl': n_people})
            processed_datasets.append(ds)

        combined_dataset = concatenate_datasets(processed_datasets)
        return combined_dataset

    train_dataset = merge_kk_dataset(train_dataset)
    test_dataset = merge_kk_dataset(test_dataset)


    def make_map_fn(split):
        def process_fn(example, idx):
            # question = make_prefix(example, template_type=args.template_type)
            question = example.pop('quiz')

            solution = {
                "solution_text_format": example['solution_text_format'],
                "statements": example['statements']
            }


            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "logic",
                'difficulty': example['n_ppl'],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    # Create local directory if not exists
    os.makedirs(os.path.expanduser(local_dir), exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
