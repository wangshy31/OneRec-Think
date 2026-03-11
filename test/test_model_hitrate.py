#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-stage model (stage1 + stage2) hit rate evaluation script with beam search and constrained generation
Supports parquet file data loading and comprehensive evaluation metrics
"""

import argparse
import json
import os
import sys
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
import random
import datetime
import numpy as np
import re
from collections import defaultdict
from typing import List, Dict, Any, Callable


def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage Model Hit Rate Test with Beam Search")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--merged_model_path", type=str,
                        default="/llm-reco-ssd-share/zhangrongzhou/Qwen3/merged_beauty_model_1-2",
                        help="Merged model path (base + stage1 + stage2)")
    parser.add_argument("--additional_lora_path", type=str, default=None,
                        help="Optional additional LoRA path to load on top of merged model")

    parser.add_argument("--test_parquet_file", type=str, 
                        default="../data/training_prediction_sid_data_test.parquet",
                        help="Test parquet file path")

    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    parser.add_argument("--num_beams", type=int, default=20, help="Number of beams for beam search")
    parser.add_argument("--sample_num", type=int, default=-1,
                        help="test sample number, -1 represents using all test data")
    parser.add_argument("--sample_offset", type=int, default=0,
                        help="sample offset for multi-GPU parallel processing")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for logging purposes")
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10",
                        help="test metrics, separate by comma")
    parser.add_argument("--filter_items", action="store_true", default=False,
                        help="whether filter illegal items")

    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p for generation")

    parser.add_argument("--enable_cot", action="store_true", default=False,
                        help="enable two-stage generation: Think then constrained Response")
    parser.add_argument("--think_max_tokens", type=int, default=64,
                        help="max new tokens for the Think stage")
    parser.add_argument("--print_generations", action="store_true", default=False,
                        help="print prompts, think, and response candidates")

    parser.add_argument("--log_file", type=str,
                        default="./logs/two_stage_test.log",
                        help="all output log file path")
    parser.add_argument("--global_trie_file", type=str, default=None,
                        help="Pre-computed global trie file for parallel evaluation")
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def setup_logging(log_file):
    """Setup logging to file"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('two_stage_test')
    logger.setLevel(logging.DEBUG)
    
    if logger.handlers:
        logger.handlers.clear()
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def format_chat_prompt(user_content):
    """Format input as chat format prompt"""
    system_message = "You are a professional recommendation expert who needs to recommend the next possible purchase for users based on their purchase history. Please predict the most likely next product that the user will purchase based on the user's historical purchase information."
    
    chat_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>

</think>
"""
    return chat_prompt


def load_merged_model(model_path, additional_lora_path=None, logger=None):
    """Load pre-merged model and tokenizer, optionally with additional LoRA"""
    logger.info(f"Loading merged model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Set left padding for generation
    
    # Force GPU usage - direct approach without device_map
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        logger.info(f"🔧 Forcing model to GPU: {device}")
        
        # Direct load and move approach (most reliable)
        logger.info("Loading model and moving to GPU...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Force move to GPU
        logger.info(f"Moving model to {device}...")
        model = model.to(device)
        logger.info(f"✅ Model moved to GPU")
        
        # Verify GPU placement
        first_param_device = next(model.parameters()).device
        if 'cuda' in str(first_param_device):
            logger.info(f"✅ Confirmed: Model is on {first_param_device}")
        else:
            logger.error(f"❌ Failed: Model is still on {first_param_device}")
            raise RuntimeError("Failed to move model to GPU")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32
        )
    
    logger.info(f"Merged model loaded successfully, tokenizer vocab size: {tokenizer.vocab_size}")
    
    # Debug: Check model device placement
    logger.info(f"🔍 Model device info:")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"  CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"  CUDA device name: {torch.cuda.get_device_name()}")
    
    # Check model device (proper way for device_map models)
    if hasattr(model, 'hf_device_map'):
        logger.info(f"  Model device map: {model.hf_device_map}")
        # Check if model is actually on GPU
        first_param = next(model.parameters())
        actual_device = first_param.device
        logger.info(f"  Model parameters actual device: {actual_device}")
        if 'cpu' in str(actual_device):
            logger.error("❌ MODEL IS STILL ON CPU! Need to fix this!")
        else:
            logger.info(f"✅ Model is correctly on GPU: {actual_device}")
    else:
        first_param = next(model.parameters())
        logger.info(f"  Model parameters device: {first_param.device}")
    
    # Optionally load additional LoRA
    if additional_lora_path and os.path.exists(additional_lora_path):
        logger.info(f"Loading additional LoRA from: {additional_lora_path}")
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, additional_lora_path)
            logger.info("Additional LoRA loaded successfully")
        except Exception as e:
            logger.error(f"Error loading additional LoRA: {e}")
            logger.info("Continuing with merged model only")
    elif additional_lora_path:
        logger.warning(f"Additional LoRA path does not exist: {additional_lora_path}")
        logger.info("Continuing with merged model only")
    
    return model, tokenizer


class ParquetTestDataset(Dataset):
    """Dataset for loading test data from parquet files"""
    
    def __init__(self, parquet_file, sample_num=-1, sample_offset=0, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Loading test data from parquet file: {parquet_file}")
        
        # Load parquet file
        self.df = pd.read_parquet(parquet_file)
        self.logger.info(f"Loaded {len(self.df)} samples from parquet")
        
        # Apply offset and sample data for multi-GPU processing
        if sample_offset > 0:
            self.df = self.df.iloc[sample_offset:].reset_index(drop=True)
            self.logger.info(f"Applied offset {sample_offset}, remaining samples: {len(self.df)}")
        
        if sample_num > 0 and sample_num < len(self.df):
            self.df = self.df.iloc[:sample_num].reset_index(drop=True)
            self.logger.info(f"Limited to {sample_num} samples for this GPU")
        
        # Expected columns: ['description', 'groundtruth', 'user_id']
        required_cols = ['description', 'groundtruth']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Required column '{col}' not found in parquet file. Available: {list(self.df.columns)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'input_ids': row['description'],
            'labels': row['groundtruth'],
            'user_id': row.get('user_id', f'user_{idx}')
        }
    
    def get_prefix_allowed_tokens_fn(self, tokenizer, global_trie_file=None):
        """Create prefix allowed tokens function for SID constrained generation based on all items in test set.
        
        FIX: Uses a proper prefix trie instead of position-based bigram table.
        The old implementation used trie[pos][token] -> {next_tokens}, which only
        considered the position and last token to determine allowed continuations.
        This ignored the full generation path and could produce invalid SID
        combinations. The new implementation walks the trie along the complete
        generated token sequence to ensure only valid SIDs are produced.
        """
        
        if not global_trie_file:
            raise ValueError("Global trie file path must be provided")
        
        if not os.path.exists(global_trie_file):
            raise FileNotFoundError(f"Global trie file not found: {global_trie_file}. Please run precompute_global_trie.py first.")
        
        # Load pre-computed prefix trie
        self.logger.info(f"Loading pre-computed prefix trie from: {global_trie_file}")
        import pickle
        with open(global_trie_file, 'rb') as f:
            trie_data = pickle.load(f)
        
        # Verify this is a prefix trie
        trie_type = trie_data.get('trie_type', None)
        if trie_type != 'prefix':
            raise ValueError(f"Expected prefix trie file, but got trie_type='{trie_type}'. Please regenerate the trie file with the updated precompute_global_trie.py.")
        
        # Load prefix trie structure
        prefix_trie = trie_data['prefix_trie']
        valid_sids = trie_data['valid_sids']
        search_space_size = trie_data.get('search_space_size', 0)
        max_length = trie_data.get('max_length', 0)
        
        self.logger.info(f"Loaded prefix trie:")
        self.logger.info(f"  Total unique SIDs: {len(valid_sids)}")
        self.logger.info(f"  Search space size: {search_space_size:,} (exact match only)")
        self.logger.info(f"  Trie depth: {max_length}")
        self.logger.info(f"  Root children (first-level tokens): {len(prefix_trie['c'])}")
        
        # Get "</think>" separator - looking for the end of think block
        sep = tokenizer("</think>", add_special_tokens=False)["input_ids"]
        
        def find_last_sublist(lst, sub):
            """Find the last occurrence of sublist in list"""
            if not sub:
                return None
            n, m = len(lst), len(sub)
            for start in range(n - m, -1, -1):
                if lst[start:start + m] == sub:
                    return start
            return None
        
        def prefix_allowed_tokens_fn(batch_id, sentence):
            """Return allowed tokens based on current generation position using prefix trie.
            
            Walks the trie along the full generated SID token path to determine
            valid continuations, ensuring only actually valid SIDs can be produced.
            """
            sentence = sentence.tolist()
            
            # Find "</think>" position
            pos = find_last_sublist(sentence, sep)
            if pos is None:
                # Before "</think>", allow all tokens
                return list(tokenizer.get_vocab().values())
            
            # Calculate position after "</think>"
            pos_after_sep = pos + len(sep)
            generated_after_sep = sentence[pos_after_sep:]
            
            # Determine current position
            current_pos = len(generated_after_sep)
            
            # Handle newline after </think> then SID pattern
            if current_pos == 0:
                # First token after </think> should be newline
                newline_tokens = tokenizer.encode('\n', add_special_tokens=False)
                return newline_tokens
            else:
                # After newline, apply SID constraints using prefix trie
                # Skip the newline token; SID tokens start from index 1
                sid_tokens = generated_after_sep[1:]
                
                # Walk the prefix trie following the actual generated path
                node = prefix_trie
                for tok in sid_tokens:
                    if tok in node['c']:
                        node = node['c'][tok]
                    else:
                        # Generated path is invalid; force EOS
                        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                        return [eos_id]
                
                # Return all valid continuations from this node
                if node['c']:
                    allowed = list(node['c'].keys())
                    # If this node also marks end of a valid SID, allow EOS too
                    if node['e']:
                        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                        if eos_id not in allowed:
                            allowed.append(eos_id)
                    return allowed
                else:
                    # Leaf node: SID is complete, return EOS
                    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
                    return [eos_id]
        
        return prefix_allowed_tokens_fn


class TestCollator:
    """Collator for test data"""
    
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"
    
    def __call__(self, batch):
        batch_prompts = []
        targets = [d["labels"] for d in batch]
        
        for d in batch:
            message = d["input_ids"]
            # Format as chat prompt
            prompt_text = format_chat_prompt(message)
            batch_prompts.append(prompt_text)
        
        return {
            "inputs": batch_prompts,
            "targets": targets
        }


def extract_sid_from_text(text):
    """Extract SID part from text, return only the SID tokens"""
    import re
    # Pattern to match SID: <|sid_begin|><s_a_X><s_b_X><s_c_X><s_d_X><|sid_end|>
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    match = re.search(sid_pattern, text)
    if match:
        return match.group(0)
    return text.strip()

def extract_all_sids_from_text(text):
    """Extract all SID tokens from text, return a list of SID strings"""
    import re
    # Pattern to match SID: <|sid_begin|><s_a_X><s_b_X><s_c_X><s_d_X><|sid_end|>
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    matches = re.findall(sid_pattern, text)
    return matches

def get_topk_results(predictions, scores, targets, k, all_items=None):
    """Extract top-k results from predictions"""
    results = []
    B = len(targets)
    predictions = [_.split("</think>")[-1] for _ in predictions]
    predictions = [_.strip().replace(" ", "") for _ in predictions]
    
    # Extract only SID parts from both predictions and targets
    predictions = [extract_sid_from_text(pred) for pred in predictions]
    
    if all_items is not None:
        for i, seq in enumerate(predictions):
            if seq not in all_items:
                scores[i] = -1000
    
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        pairs = [(seq, score) for seq, score in zip(batch_seqs, batch_scores)]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        
        # Extract SID from target as well
        target_item = extract_sid_from_text(targets[b])
        
        one_results = []
        for pred_seq, pred_score in sorted_pairs:
            if pred_seq == target_item:
                one_results.append(1)
            else:
                one_results.append(0)
        
        results.append(one_results)
    
    return results


def hit_k(topk_results, k):
    """Calculate hit@k metric"""
    hit = 0.0
    for row in topk_results:
        if len(row) >= k and max(row[:k]) == 1:
            hit += 1
    return hit / len(topk_results)


def ndcg_k(topk_results, k):
    """Calculate ndcg@k metric"""
    ndcg = 0.0
    for row in topk_results:
        dcg = 0.0
        for i in range(min(k, len(row))):
            if row[i] == 1:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 1.0 / np.log2(2)  # Best case: hit at position 1
        ndcg += dcg / idcg
    return ndcg / len(topk_results)


def get_metrics_results(topk_results, metrics):
    """Calculate evaluation metrics"""
    res = {}
    for m in metrics:
        if m.lower().startswith("hit"):
            k = int(m.split("@")[1])
            res[m] = hit_k(topk_results, k)
        elif m.lower().startswith("ndcg"):
            k = int(m.split("@")[1])
            res[m] = ndcg_k(topk_results, k)
        else:
            raise NotImplementedError(f"Metric {m} not implemented")
    
    return res


def extract_assistant_response(generated_text):
    """Extract assistant response from generated text"""
    # Try to extract content after </think>
    if "</think>" in generated_text:
        response_part = generated_text.split("</think>")[-1].strip()
        # Extract only the SID part using the existing function
        return extract_sid_from_text(response_part)
    
    # Fallback to assistant pattern
    if "<|im_start|>assistant" in generated_text:
        parts = generated_text.split("<|im_start|>assistant")
        if len(parts) > 1:
            assistant_response = parts[1]
            if "<|im_end|>" in assistant_response:
                assistant_response = assistant_response.split("<|im_end|>")[0]
            # Extract only the SID part from the assistant response
            return extract_sid_from_text(assistant_response.strip())
    
    # Final fallback - try to extract SID from the entire text
    return extract_sid_from_text(generated_text)


def run_evaluation(args):
    """Main evaluation function"""
    set_seed(args.seed)
    logger = setup_logging(args.log_file)
    logger.info(f"🚀 Starting Two-stage Model Hit Rate Evaluation [GPU {args.gpu_id}]")
    logger.info(f"Args: {vars(args)}")
    
    # 1. Load merged model
    logger.info("=" * 60)
    logger.info("Loading merged model...")
    final_model, tokenizer = load_merged_model(
        args.merged_model_path, 
        args.additional_lora_path, 
        logger
    )
    final_model.eval()
    
    # 2. Load test dataset
    logger.info("📊 Loading test dataset...")
    if not os.path.exists(args.test_parquet_file):
        raise FileNotFoundError(f"Parquet file not found: {args.test_parquet_file}")
    
    test_dataset = ParquetTestDataset(args.test_parquet_file, args.sample_num, args.sample_offset, logger)
    prefix_allowed_tokens_fn = test_dataset.get_prefix_allowed_tokens_fn(tokenizer, args.global_trie_file)
    logger.info(f"Using parquet file: {args.test_parquet_file}")
    if args.global_trie_file:
        logger.info(f"✅ Global trie file: {args.global_trie_file}")
    logger.info("✅ SID constrained generation enabled")
    
    collator = TestCollator(args, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=0,  # Use 0 for compatibility
        pin_memory=True
    )
    
    logger.info(f"📈 Test data size: {len(test_dataset)}")
    
    # 3. Start evaluation
    metrics = args.metrics.split(",")
    all_topk_results = []  # 累积所有样本的topk结果
    total = 0
    
    logger.info("🚀 Starting evaluation...")
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Testing")
        for step, batch in enumerate(progress_bar):
            inputs_texts = batch["inputs"]
            targets = batch["targets"]
            bs = len(targets)
            
            # Calculate progress information
            current_step = step + 1
            total_steps = len(test_loader)
            elapsed = time.time() - start_time
            if current_step > 0:
                avg_time = elapsed / current_step
                remaining_time = avg_time * (total_steps - current_step)
                
                # Format times
                elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
                remaining_str = f"{int(remaining_time//60):02d}:{int(remaining_time%60):02d}"
                
                # Create progress bar visual
                progress_pct = current_step / total_steps
                bar_length = 10
                filled = int(progress_pct * bar_length)
                bar = '█' * filled + '░' * (bar_length - filled)
                
                progress_info = f"Testing: {progress_pct*100:3.0f}%|{bar}| {current_step}/{total_steps} [{elapsed_str}<{remaining_str}, {avg_time:.2f}s/it]"
                logger.info(progress_info)
            
            # === Skip CoT Think stage - generate SID directly ===
            think_texts = [""] * bs
            logger.info(f"🚀 Skipping CoT Think stage - generating SID directly for batch {step}...")

            # === Generate SID directly (no CoT, no Response: prefix) ===
            # Use the formatted prompt as-is, which ends with </think>\n
            response_inputs_texts = inputs_texts
            
            # Encode inputs
            enc = tokenizer(
                response_inputs_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            )
            enc = {k: v.to(final_model.device) for k, v in enc.items()}
            
            # Debug: Check tensor devices in Response stage  
            logger.info(f"🔍 Response stage device info:")
            logger.info(f"  Input tensor device: {enc['input_ids'].device}")
            logger.info(f"  Model device: {next(final_model.parameters()).device}")
            
            # Beam search generation
            num_beams = args.num_beams
            while True:
                try:
                    generate_kwargs = {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc.get("attention_mask", None),
                        "max_new_tokens": args.max_new_tokens,
                        "num_beams": num_beams,
                        "num_return_sequences": num_beams,
                        "output_scores": True,
                        "return_dict_in_generate": True,
                        "early_stopping": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    }
                    
                    # Add SID constrained generation
                    if prefix_allowed_tokens_fn is not None:
                        generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed_tokens_fn
                    
                    output = final_model.generate(**generate_kwargs)
                    break
                except RuntimeError as e:
                    err = str(e).lower()
                    if "out of memory" in err or "cuda" in err:
                        logger.warning(f"CUDA OOM with beam={num_beams}. Reducing beam size.")
                        num_beams -= 1
                        if num_beams < 1:
                            raise RuntimeError("Beam search OOM even with beam=1") from e
                        torch.cuda.empty_cache()
                    else:
                        raise
            
            # Decode output
            output_ids = output["sequences"]
            scores = output.get("sequences_scores", None)
            decoded = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            
            # Process scores (always needed for metrics calculation)
            if scores is not None:
                if hasattr(scores, 'detach'):
                    scores_list = [float(s) for s in scores.detach().cpu().tolist()]
                else:
                    scores_list = [float(s) for s in scores]
            else:
                scores_list = [0.0] * len(decoded)
            
            # Print generations if requested
            if args.print_generations:
                for i in range(bs):
                    start = i * num_beams
                    end = start + num_beams
                    cands = decoded[start:end]
                    cand_scores = scores_list[start:end]
                    
                    logger.info(f"----- SAMPLE {step*bs + i} -----")
                    if args.enable_cot and think_texts[i]:
                        logger.info(f"THINK: {think_texts[i]}")
                    
                    # Show the complete prompt
                    logger.info(f"PROMPT: {inputs_texts[i]}")
                    
                    logger.info("RESPONSE_CANDIDATES:")
                    for j, (c, sc) in enumerate(zip(cands, cand_scores)):
                        response = extract_assistant_response(c)
                        logger.info(f"  Rank {j+1}: score={sc:.4f} → {response}")
                    logger.info(f"TARGET: {targets[i]}")
                    logger.info("-" * 50)
            
            # Calculate topk results (no additional filtering needed since exact trie already constrains generation)
            topk_res = get_topk_results(
                decoded, scores_list, 
                targets, num_beams,
                all_items=None
            )
            
            # Accumulate all topk results (extend instead of sum)
            all_topk_results.extend(topk_res)
            total += bs
            
            # Progress report every 50 steps
            if (step + 1) % 50 == 0:
                # Calculate metrics on accumulated results so far
                temp_metrics_results = get_metrics_results(all_topk_results, metrics)
                logger.info("=" * 50)
                logger.info(f"📊 PROGRESS REPORT - Step {step+1}/{len(test_loader)}")
                logger.info(f"💾 Processed samples: {total}")
                logger.info("📈 Current Metrics:")
                for metric, value in temp_metrics_results.items():
                    logger.info(f"  {metric:>10}: {value:.4f}")
                logger.info("=" * 50)
    
    # 4. Final results - calculate metrics on all accumulated results
    final_metrics_results = get_metrics_results(all_topk_results, metrics)
    
    logger.info("=" * 60)
    logger.info("🎯 Final Hit Rate Results:")
    logger.info("=" * 60)
    for metric, value in final_metrics_results.items():
        logger.info(f"{metric:>10}: {value:.4f}")
    logger.info("=" * 60)
    
    # 5. Test summary
    logger.info("\n📊 Test Summary:")
    logger.info(f"Merged model: {args.merged_model_path}")
    if args.additional_lora_path:
        logger.info(f"Additional LoRA: {args.additional_lora_path}")
    logger.info(f"Parquet file: {args.test_parquet_file}")
    logger.info(f"Total samples: {total}")
    logger.info(f"Batch size: {args.test_batch_size}")
    logger.info(f"Beam size: {args.num_beams}")
    logger.info(f"CoT enabled: {args.enable_cot}")
    if args.enable_cot:
        logger.info(f"Think max tokens: {args.think_max_tokens}")
    
    logger.info("\n✅ Evaluation completed successfully!")
    
    return final_metrics_results


def main():
    """Main function"""
    args = parse_args()
    
    try:
        results = run_evaluation(args)
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
