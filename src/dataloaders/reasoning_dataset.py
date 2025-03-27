"""
Generic dataset loader that supports simple input/output format
"""
from functools import partial
import os
from os.path import join
from transformers import PreTrainedTokenizer
from datasets import load_metric, load_dataset

from .utils import (
    get_lm_loader, get_seq2seq_loader,
    convert_to_hf_dataset, 
    get_tokenizer_from_config,
    download_scrolls_metric as download_metric
)
from .utils.packing import ConcatDataset


def load_data(name: str, dataset_config: dict, pretrained_model_config: dict,
              preprocess_config: dict, **loader_kwargs: any):
    """
    Load dataset in a generic input/output format
    """
    # Misc. setup
    cache_dir = dataset_config['cache_dir']
    input_len = dataset_config['chunk_size']
    n_elms = dataset_config.pop('n_elms', None)
    concat_data = dataset_config['concat_data']

    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]
    
    # Setup tokenizer
    tokenizer = get_tokenizer_from_config(pretrained_model_config)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation

    with open(os.path.join('reasoning_template.jinja'), 'r') as f:
        template = f.read()
    tokenizer.chat_template = template
    # Get initial data
    ignore_kwargs = ['concat_data', 'chunk_size']
    dataset = load_dataset(
        **{k: v for k, v in dataset_config.items() if k not in ignore_kwargs}
    )

   
    # If no splits defined, create them from the main dataset
    dataset = dataset['train']
    if n_elms is not None:
        dataset = dataset.shuffle(seed=42).select(range(n_elms))
    pct_train = 0.8
    pct_val = 0.01
    
    train_set = convert_to_hf_dataset(dataset.select(range(int(len(dataset) * pct_train))), cache_dir)
    val_set = convert_to_hf_dataset(dataset.select(range(int(len(dataset) * pct_train), int(len(dataset) * (pct_train + pct_val)))), cache_dir)
    test_set = convert_to_hf_dataset(dataset.select(range(int(len(dataset) * (pct_train + pct_val)), len(dataset))), cache_dir)

    # Convert to dicts of {input_ids, attention_mask, labels}
    train_set = train_set.map(
        partial(tokenize_sample, tokenizer=tokenizer, include_label=True),
        remove_columns=list(dataset.features),
    )
    val_set = val_set.map(
        partial(tokenize_sample, tokenizer=tokenizer, include_label=True),
        remove_columns=list(dataset.features),
    )   
    test_set = test_set.map(
        partial(tokenize_sample, tokenizer=tokenizer, include_label=False),
        remove_columns=list(dataset.features),
    )

    # Chunk together train and val sets
    if concat_data:
        train_set = ConcatDataset(train_set, chunk_size=input_len)
        val_set = ConcatDataset(val_set, chunk_size=input_len)
        print("train_set post concat", len(train_set))
        print("train_set[0] post concat", train_set[0])
        
    
    # Get dataloaders
    dataloaders = {
        'train': get_lm_loader(train_set, tokenizer, 'train', input_len, **loader_kwargs),
        'validation': get_lm_loader(val_set, tokenizer, 'validation', input_len, **loader_kwargs),
        'test': get_seq2seq_loader(test_set, tokenizer, 'test', **loader_kwargs),
    }

    # Evaluation metric
    try:
        metric = load_metric(download_metric(), 'gov_report')  # hack but we want rouge
    except Exception as e:
        print(f'Error loading metric: {e}')
        metric = None

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
        dataloaders[k].dataset.metric = metric
    return dataloaders


def tokenize_sample(sample, tokenizer : PreTrainedTokenizer, include_label: bool = True):
    """
    Tokenize input and output fields into a single sequence
    """
    import os
    with open(os.path.join('reasoning_template.jinja'), 'r') as f:
        template = f.read()
    # use the first idx that is True
    max_idx = max([i for i, v in enumerate(sample['correctness_math_verify']) if v]+[0])
    generation = sample['generations'][max_idx]
    
    assert sample['messages'][1]['role'] == 'assistant'
    # we need to replace the assistant message with the generation
    sample['messages'][1]['content'] = generation
    
    dict_out = tokenizer.apply_chat_template(
        sample['messages'],
        return_tensors="pt",
        chat_template=template,
        return_dict=True,
    )
    
    return {
        'input_ids': dict_out['input_ids'][0],
        'attention_mask': dict_out['attention_mask'][0],
        'labels': dict_out['input_ids'][0],
    }