"""
Evaluation script for AIME dataset using vLLM for faster inference with base model
"""
import argparse
import json
import re
import os
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from typing import List
from vllm import LLM, SamplingParams
from omegaconf import OmegaConf
import os

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
from src.utils.setup import seed_everything
from src.utils.logging import print_header
from src.model.pretrained import get_pretrained_loader

def extract_boxed_answer(text: str) -> str:
    """Extract the answer from \boxed{} command in the text."""
    match = re.search(r'\\boxed{([^}]+)}', text)
    if match:
        return match.group(1)
    return None

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--model_config", type=str, default="configs/model/base_llama3_1_8b.yaml")
    parser.add_argument("--config_dir", type=str, default="configs")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--n_parallel", type=int, default=5)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    seed_everything(args.seed)
    
    # Load model config
    model_config = OmegaConf.load(args.model_config)
    
    # Initialize vLLM model
    print_header("Loading model with vLLM...")
    
    
    # Load AIME dataset
    print_header("Loading AIME dataset...")
    ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split='train')
    if args.year != -1:
        print(f"Filtering dataset for year {args.year}, len pre-filter: {len(ds)}")
        ds = ds.filter(lambda x: x['Year'] == args.year)
        print(f"Loaded {len(ds)} problems from year {args.year}")
    
    # Initialize metrics
    total_correct = 0
    total_samples = 0
    model_name = model_config.model.pretrained_model_name_or_path.split('/')[-1]
    
    traces_path = f"reasoning_traces/{model_name}_vllm"
    os.makedirs(traces_path, exist_ok=True)
    
    print_header(f"Starting evaluation with {model_name}")
    llm = LLM(
        model=model_config.model.pretrained_model_name_or_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        seed=args.seed,
    )
    # Process examples
    n_samples = min(args.n_samples, len(ds))
    
    for samples in ds.iter(batch_size=args.n_parallel):
        # Create prompt
        samples = [{k:v[i] for k,v in samples.items()} for i in range(len(samples['Question']))]
        prompts = [f"Question: {sample['Question']}\nAnswer: <think>" for sample in samples]
        print(prompts)
        
        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
        )
        
        # Generate response
        outputs = llm.generate(prompts, sampling_params)
        generated_texts = [output.text for output in outputs[0].outputs]
        
        # Process results
        predicted_answers = [extract_boxed_answer(text) for text in generated_texts]
        
        for i, (generated_text, predicted_answer) in enumerate(zip(generated_texts, predicted_answers)):
            if predicted_answer is not None:
                total_samples += 1
                is_correct = predicted_answer.strip() == samples[i]['Answer'].strip()
                if is_correct:
                    total_correct += 1
            else:
                is_correct = False
            
            # Save result
            result_dict = {
                "problem_number": samples[i]['Problem Number'],
                "question": samples[i]['Question'],
                "generated_text": generated_text,
                "predicted_answer": predicted_answer,
                "actual_answer": samples[i]['Answer'],
                "correct": is_correct
            }
            
            # Save individual result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'{traces_path}/problem_{samples[i]["Problem Number"]}_{timestamp}.json', "w") as f:
                json.dump(result_dict, f, indent=2)

    # Print final results
    print_header("Evaluation Results")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Correct answers: {total_correct}")
    print(f"Accuracy: {(total_correct/total_samples)*100:.2f}%")

if __name__ == '__main__':
    main() 