import os
import datasets
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/curriculum-llm/data/math-difficulty')

    args = parser.parse_args()

    data_source = 'pe-nlp/ORZ-MATH-57k-Filter-difficulty'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, split='train', trust_remote_code=True)    

    split_dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn(split):

        def process_fn(example, idx):
            question = example.pop('problem')

            question = question + ' ' + instruction_following

            answer = example.pop('answer')

            pass_at_n = float(example.pop('pass_at_n'))

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "difficulty": int((1.0 - pass_at_n) * 10),
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn


    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))



