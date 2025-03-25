from datasets import Dataset, load_dataset, DatasetDict
import os
from openai import OpenAI
import argparse
import re
from verl.utils.reward_score.gsm8k import compute_score
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split('#### ')[1].replace(',', '')
    return final_solution

# 创建线程局部存储的客户端
thread_local = threading.local()

def get_client(args):
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=os.environ.get("ARK_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            timeout=args.timeout,
        )
    return thread_local.client

def get_model_output(args, question):
    instruction_following = "Let's think step by step and output the final answer after \"####\"."
    client = get_client(args)
    response = client.chat.completions.create(
        model=args.model_name,
        messages=[
            {"role": "user", "content": question + ' ' + instruction_following}
        ],
        temperature=args.temperature,
    )
    return response.choices[0].message.content

def process_sample(sample, args):
    question = sample['question']
    answer = sample['answer']
    ground_truth = extract_solution(answer)
    
    # 并行处理每个样本的多次尝试
    pass_count = 0
    with ThreadPoolExecutor(max_workers=args.sample_n) as executor:
        future_to_response = {
            executor.submit(get_model_output, args, question): i 
            for i in range(args.sample_n)
        }
        
        for future in as_completed(future_to_response):
            response = future.result()
            res = compute_score(response, ground_truth)
            if res == 1.0:
                pass_count += 1
    
    return {
        'question': question,
        'answer': answer,
        'ground_truth': ground_truth,
        'pass_at_n': pass_count/args.sample_n,
        'sample_n': args.sample_n,
    }


def process_dataset(dataset, args, split_name):
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_sample = {executor.submit(process_sample, sample, args): sample 
                          for sample in dataset}
        
        for future in tqdm(as_completed(future_to_sample), total=len(dataset), desc=f"Processing {split_name}"):
            result = future.result()
            results.append(result)
            print(f"question: {result['question']}  pass_at_n: {result['pass_count']}")
    
    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='doubao-1-5-lite-32k-250115')
    parser.add_argument('--sample_n', default=8, type=int)
    parser.add_argument('--local_dir', default='~/curriculum-llm/raw_data/gsm8k-pass-at-n')
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--timeout', default=1000, type=int)
    parser.add_argument('--max_workers', default=32, type=int)  # 添加控制并行度的参数

    args = parser.parse_args()

    dataset = load_dataset('openai/gsm8k', 'main')
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # test logic
    # train_dataset = train_dataset.select(range(8))
    # test_dataset = test_dataset.select(range(3))

    train_results = process_dataset(train_dataset, args, "train")
    test_results = process_dataset(test_dataset, args, "test")

    results_dataset = DatasetDict({
        'train': Dataset.from_list(train_results),
        'test': Dataset.from_list(test_results)
    })
    

    output_dir = os.path.expanduser(args.local_dir)
    os.makedirs(output_dir, exist_ok=True)
    results_dataset.save_to_disk(output_dir)

    print(f"Dataset saved to {output_dir} with 'train' and 'test' splits")
