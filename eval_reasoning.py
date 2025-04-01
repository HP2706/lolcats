"""
Evaluation script for AIME dataset using LoLCATs model
"""
import argparse
import json
import torch
import re
import os
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
from typing import List

from src.utils.setup import seed_everything
from src.utils.logging import print_header

# Import functions from demo_lolcats_llm.py
from demo_lolcats_llm import (
    get_args as get_demo_args,
    load_model_from_checkpoint,
    get_model_name,
    system_prompt,
    BatchTextIteratorStreamer
)

def extract_boxed_answer(text: str) -> str:
    """Extract the answer from \boxed{} command in the text."""
    match = re.search(r'\\boxed{([^}]+)}', text)
    if match:
        return match.group(1)
    return None

def get_args():
    # Get all arguments from demo script
    parser = get_demo_args(parse=False)

    # Add evaluation specific arguments
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--n_parallel", type=int, default=1)
    
    args = parser.parse_args()
    return args

class FileWriterStreamer(BatchTextIteratorStreamer):
    """Streamer that writes each batch generation to separate files"""
    def __init__(self, 
                 tokenizer, 
                 batch_size: int,
                 output_dir: str,
                 problem_numbers: list,
                 skip_prompt: bool = False, 
                 **decode_kwargs):
        super().__init__(tokenizer, batch_size, skip_prompt, **decode_kwargs)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create file handles for each batch item
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.files = [
            open(f"{output_dir}/problem_{num}_{timestamp}.txt", "w", buffering=1)  # Line buffering
            for num in problem_numbers
        ]
    
    def on_finalized_text(self, texts: List[str], stream_end: bool = False):
        # Write each batch item's text to its corresponding file
        for text, f in zip(texts, self.files):
            f.write(text)
            f.flush()  # Ensure writing to disk
        
        if stream_end:
            # Close all files when generation is complete
            for f in self.files:
                f.close()

def main():
    args = get_args()
    seed_everything(args.seed)
    
    # Load model and tokenizer
    model, model_config, tokenizer = load_model_from_checkpoint(
        args.attn_mlp_checkpoint_path, args.finetune_checkpoint_path, 
        config_dir=args.config_dir, print_model=args.print_model, debug=args.debug,
        huggingface_token=args.huggingface_token, use_cuda_kernels=args.use_cuda_kernels
    )
    model.eval()
    
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
    model_name = get_model_name(args.attn_mlp_checkpoint_path, 
                              args.finetune_checkpoint_path, 
                              model_config)
    
    traces_path = f"reasoning_traces/{model_name}"
    os.makedirs(traces_path, exist_ok=True)
    
    print_header(f"Starting evaluation with {model_name}")
    
    # Process examples in batches
    batch_size = args.n_parallel  # Using n_parallel for batch size
    n_samples = min(args.n_samples, len(ds))
    
    for batch_start in tqdm(range(0, n_samples, batch_size)):
        batch_end = min(batch_start + batch_size, n_samples)
        batch_samples = [ds[i] for i in range(batch_start, batch_end)]
        
        with torch.no_grad():
            # Create batch of inputs
            model_inputs = tokenizer.apply_chat_template(
                [
                    [
                        {"role": "user", "content": sample['Question']},
                        {"role": "assistant", "content": "<think>"}
                    ]
                    for sample in batch_samples
                ],
                return_dict=True,
                return_tensors="pt",
                padding=True
            )
            model_inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in model_inputs.items()}
            
            # Setup streamer for batch processing with file writing
            streamer = FileWriterStreamer(
                tokenizer=tokenizer,
                batch_size=len(batch_samples),
                output_dir=traces_path,
                problem_numbers=[sample['Problem Number'] for sample in batch_samples],
                skip_prompt=True,
            )
            
            # Generate for entire batch
            model_outputs = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=args.top_k,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
            
            # Process results for metrics
            generated_texts = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
            
            # Update metrics and save final results
            results = []
            for i, (sample, generated_text) in enumerate(zip(batch_samples, generated_texts)):
                predicted_answer = extract_boxed_answer(generated_text)
                
                if predicted_answer is not None:
                    total_samples += 1
                    is_correct = predicted_answer.strip() == sample['Answer'].strip()
                    if is_correct:
                        total_correct += 1
                    
                    result_dict = {
                        "problem_number": sample['Problem Number'],
                        "question": sample['Question'],
                        "generated_text": generated_text,
                        "predicted_answer": predicted_answer,
                        "actual_answer": sample['Answer'],
                        "correct": is_correct
                    }
                    results.append(result_dict)
            
            # Save batch results to JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'{traces_path}/batch_{batch_start}_{timestamp}_results.json', "w") as f:
                json.dump(results, f, indent=2)

    # Print final results
    print_header("Evaluation Results")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Correct answers: {total_correct}")
    print(f"Accuracy: {(total_correct/total_samples)*100:.2f}%")

if __name__ == '__main__':
    main()
