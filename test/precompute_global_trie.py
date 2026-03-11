#!/usr/bin/env python3

import pandas as pd
import pickle
import argparse
import os
import re
from transformers import AutoTokenizer


def extract_all_sids_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    matches = re.findall(sid_pattern, text)
    return matches


def extract_sid_from_text(text):
    sid_pattern = r'<\|sid_begin\|><s_a_\d+><s_b_\d+><s_c_\d+><s_d_\d+><\|sid_end\|>'
    match = re.search(sid_pattern, text)
    if match:
        return match.group(0)
    return text.strip()


class TrieNode:
    """A node in the prefix trie for constrained decoding."""
    __slots__ = ['children', 'is_end']

    def __init__(self):
        self.children = {}   # token_id -> TrieNode
        self.is_end = False  # whether this node marks the end of a valid SID


def build_prefix_trie(sid_token_sequences):
    """Build a proper prefix trie from SID token sequences.
    
    Unlike the position-based bigram approach, this trie tracks the full
    prefix path so that only actually valid SID combinations are allowed
    during constrained decoding.
    """
    root = TrieNode()
    for seq in sid_token_sequences:
        node = root
        for token_id in seq:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()
            node = node.children[token_id]
        node.is_end = True
    return root


def trie_to_dict(root):
    """Serialize a TrieNode tree to a plain dict for pickling.
    
    Format: each node is {'c': {token_id: child_node_dict, ...}, 'e': bool}
    """
    result = {'c': {}, 'e': root.is_end}
    for token_id, child in root.children.items():
        result['c'][token_id] = trie_to_dict(child)
    return result


def build_global_trie(test_parquet_file, model_path, output_file):
    print(f"Loading test data from: {test_parquet_file}")
    df = pd.read_parquet(test_parquet_file)
    print(f"Total samples in test set: {len(df)}")

    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Extracting all SIDs from test set (description + groundtruth)...")
    valid_sids = set()

    for _, row in df.iterrows():
        description_sids = extract_all_sids_from_text(row['description'])
        for sid in description_sids:
            if sid and '<|sid_begin|>' in sid and '<|sid_end|>' in sid:
                valid_sids.add(sid)

        groundtruth_sid = extract_sid_from_text(row['groundtruth'])
        if groundtruth_sid and '<|sid_begin|>' in groundtruth_sid and '<|sid_end|>' in groundtruth_sid:
            valid_sids.add(groundtruth_sid)
    
    print(f"Found {len(valid_sids)} unique valid SIDs in test set")

    print("Converting SIDs to token sequences...")
    sid_token_sequences = []
    for sid in valid_sids:
        tokens = tokenizer.encode(sid, add_special_tokens=False)
        sid_token_sequences.append(tokens)
    
    print(f"Converted {len(sid_token_sequences)} SIDs to token sequences")

    # ----- FIX: Build a proper prefix trie instead of position-based bigram -----
    print("Building prefix trie...")
    trie_root = build_prefix_trie(sid_token_sequences)
    prefix_trie_dict = trie_to_dict(trie_root)

    max_length = max(len(seq) for seq in sid_token_sequences) if sid_token_sequences else 0
    print(f"Maximum SID token length: {max_length}")
    
    print(f"Built prefix trie:")
    print(f"  Total unique SIDs: {len(valid_sids)}")
    print(f"  Search space size: {len(valid_sids):,} (exact match only)")
    print(f"  Trie depth: {max_length}")
    print(f"  Root children (first-level tokens): {len(prefix_trie_dict['c'])}")

    trie_data = {
        'prefix_trie': prefix_trie_dict,
        'valid_sids': valid_sids,
        'valid_sid_tokens': sid_token_sequences,
        'tokenizer_name': model_path,
        'total_samples': len(df),
        'search_space_size': len(valid_sids),
        'max_length': max_length,
        'trie_type': 'prefix'
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(trie_data, f)
    
    print(f"Prefix trie saved to: {output_file}")
    return trie_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute global trie for parallel evaluation")
    parser.add_argument("--test_parquet_file", type=str, required=True, help="Test parquet file")
    parser.add_argument("--model_path", type=str, required=True, help="Model path for tokenizer")
    parser.add_argument("--output_file", type=str, default="./global_trie.pkl", help="Output pickle file")
    
    args = parser.parse_args()
    build_global_trie(args.test_parquet_file, args.model_path, args.output_file)
