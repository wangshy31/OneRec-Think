#!/usr/bin/env python3

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, HfArgumentParser, EarlyStoppingCallback
import pandas as pd
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import random
import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="./results/beauty_sid_rec/checkpoint-8388",
        metadata={"help": "Path to pretrained model"}
    )
    data_path: Optional[str] = field(
        default="../data/training_RA_train.parquet",
        metadata={"help": "Path to training data"}
    )
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "LoRA target modules"}
    )

def prepare_chat_dataset(data_path, sample_size=None, local_rank=0):
    if local_rank == 0:
        print(f"Loading parquet file: {data_path}")
    data_pq = pd.read_parquet(data_path)
    if local_rank == 0:
        print(f"Data shape: {data_pq.shape}")
        print(f"Columns: {list(data_pq.columns)}")

    if sample_size is not None and len(data_pq) > sample_size:
        if local_rank == 0:
            print(f"Sampling {sample_size} samples from {len(data_pq)} total samples")
        data_pq = data_pq.head(sample_size)

    texts = []
    
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    
    for _, row in data_pq.iterrows():
        if('title' in row.keys() and row['title'] is not None):
            title = row['title']
            categories = row['categories']
            assistant_content = f"<think>\nThe user is likely to buy items in {categories} category\n</think>\n{row['groundtruth']}"
        else:
            assistant_content = f"<think>\n\n</think>\n{row['groundtruth']}"
        formatted_text = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{row['description']}<|im_end|>
<|im_start|>assistant
{assistant_content}<|im_end|>
"""
        texts.append(formatted_text)
    
    if local_rank == 0:
        print(f"Total texts: {len(texts)}")

        print(f"\\nFirst 3 text examples:")
        for i, text in enumerate(texts[:3]):
            print(f"  [{i}] Length: {len(text)} chars")
            print(f"  [{i}] Text: {text[:1000]}...")
            print(f"      (Note: Loss calculated from <|im_start|>user onwards)")
            print()
    
    dataset_dict = {
        'text': texts
    }
    return Dataset.from_dict(dataset_dict)


def tokenize_function(examples, tokenizer):
    tokenized = tokenizer(
        examples['text'],
        # Pad only in the collator:
        # padding added here would end up in labels (as pad_token_id) because the collator cannot tell it apart.
        padding=False,
        truncation=True,
        max_length=4096,
        add_special_tokens=True,
        return_attention_mask=True,
    )

    user_start_tokens = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
    m = len(user_start_tokens)

    all_labels = []

    for input_ids in tokenized['input_ids']:
        n = len(input_ids)
        labels = input_ids.copy()
        start = -1
        for i in range(n - m + 1):
            if input_ids[i: i + m] == user_start_tokens:
                start = i
                break

        if start != -1:
            labels[:start] = [-100] * start
        else:
            labels = [-100] * n
        all_labels.append(labels)

    tokenized['labels'] = all_labels
    return tokenized

def get_special_tokens():
    special_tokens = []

    special_tokens.append('<|sid_begin|>')
    special_tokens.append('<|sid_end|>')

    max_range = 256
    for prefix in ['s_a', 's_b', 's_c', 's_d']:
        for i in range(max_range):
            special_tokens.append(f'<{prefix}_{i}>')
    
    return special_tokens

class CustomDataCollator:
    def __init__(self, tokenizer, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        max_length = max(len(feature["input_ids"]) for feature in features)

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for feature in features:

            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            label = feature["labels"]

            pad_len = max_length - len(input_ids)
            padded_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_mask = attention_mask + [0] * pad_len
            padded_label = label + [-100] * pad_len

            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
            padded_labels.append(padded_label)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    training_args.label_names = ["labels"]

    if training_args.local_rank == 0:
        print(f"Debug: eval_strategy = {training_args.eval_strategy}")
        print(f"Debug: save_strategy = {training_args.save_strategy}")
        print(f"Debug: metric_for_best_model = {training_args.metric_for_best_model}")
        print(f"Debug: greater_is_better = {training_args.greater_is_better}")
        print(f"Debug: load_best_model_at_end = {training_args.load_best_model_at_end}")
        print(f"Debug: early stopping patience = 2")

    model_dir = model_args.model_name_or_path
    
    if training_args.local_rank == 0:
        print(f"Loading model from: {model_dir}")

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    
    if training_args.local_rank == 0:
        print(f"Model loaded successfully")
        print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    special_tokens = get_special_tokens()
    if training_args.local_rank == 0:
        print(f"Total special tokens: {len(special_tokens)}")

    tokenized_special_tokens = tokenizer.convert_tokens_to_ids(special_tokens)

    valid_special_token_ids = []
    valid_special_tokens = []
    for i, token_id in enumerate(tokenized_special_tokens):
        if token_id != tokenizer.unk_token_id:
            valid_special_token_ids.append(token_id)
            valid_special_tokens.append(special_tokens[i])
    
    if training_args.local_rank == 0:
        print(f"Valid special tokens: {len(valid_special_token_ids)}")
        print(f"First 10 valid special tokens: {valid_special_tokens[:10]}")
        print(f"Training token IDs range: {min(valid_special_token_ids)} to {max(valid_special_token_ids)}")

    if model_args.use_lora:
        target_modules = model_args.lora_target_modules.split(",")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            trainable_token_indices={
                'embed_tokens': valid_special_token_ids,
            }
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.local_rank == 0:
            print(f"\\nTrainable parameters:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}: {param.shape}")
    else:
        if training_args.local_rank == 0:
            print("Using full parameter training")
            print(f"Special token embeddings will be trained for {len(valid_special_token_ids)} tokens")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        if training_args.local_rank == 0:
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    if training_args.local_rank == 0:
        print(f"\\nLoading training dataset...")

    print(f"Loading training dataset from: {model_args.data_path}")
    train_dataset = prepare_chat_dataset(model_args.data_path, local_rank=training_args.local_rank)
    if training_args.local_rank == 0:
        print(f"Loaded raw train dataset, total samples: {len(train_dataset)}")

    if training_args.local_rank == 0:
        print(f"\\nLoading validation dataset...")
    val_data_path = '../data/training_RA_val.parquet'
    val_dataset = prepare_chat_dataset(val_data_path, local_rank=training_args.local_rank)
    if training_args.local_rank == 0:
        print(f"Loaded raw validation dataset, total samples: {len(val_dataset)}")

    if training_args.local_rank == 0:
        print("Tokenizing training dataset...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    if training_args.local_rank == 0:
        print(f"Tokenized train dataset, total samples: {len(train_dataset)}")

    if training_args.local_rank == 0:
        print("Tokenizing validation dataset...")
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )
    if training_args.local_rank == 0:
        print(f"Tokenized validation dataset, total samples: {len(val_dataset)}")

    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    if training_args.local_rank == 0:
        print(f"\\nTrainer eval_strategy: {trainer.args.eval_strategy}")
        print(f"Trainer has eval_dataset: {trainer.eval_dataset is not None}")
        print(f"Eval dataset size: {len(trainer.eval_dataset) if trainer.eval_dataset else 0}")
    
    if training_args.local_rank == 0:
        print(f"\\nStarting training...")
    trainer.train()

    if training_args.local_rank == 0:
        print(f"\\nFinal evaluation...")
    result = trainer.evaluate()
    if training_args.local_rank == 0:
        print("Final evaluation result:")
        print(result)

    if training_args.local_rank == 0:
        print(f"\\nSaving model...")
    output_dir = training_args.output_dir
    trainer.save_model(output_dir)
    if training_args.local_rank == 0:
        print(f"Model saved to: {output_dir}")
        print("Training completed!")
