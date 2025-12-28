import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import signal
import traceback

from hierarchos import (
    pick_device,
    set_threads,
    load_full_model_with_config,
    HierarchosCore,
    train,
    finetune,
    chat,
    is_directml_device,
    setup_msvc_environment,
    create_map_style_dataloader,
    HuggingFaceMapStyleDataset,
    OriginalJSONLDataset,
    process_text_sample,
    create_dataloader_for_chunked,
    create_dataloader_pt_chunked
)

def main():
    parser = argparse.ArgumentParser(description="hierarchos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora"], help="Operation mode.")

    # --- Data and Path Arguments ---
    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, nargs='?', default=None, const=True, help="Path to local training data.")
    path_group.add_argument("--hf_dataset", type=str, default=None, help="Name or path to a Hugging Face dataset.")
    path_group.add_argument("--hf_dataset_config", type=str, default=None, help="Optional configuration name for the HF dataset.")
    path_group.add_argument("--hf_dataset_split", type=str, default="train", help="Dataset split to use.")
    path_group.add_argument("--text_column", type=str, default=None, help="Column name for text completion data.")
    path_group.add_argument("--prompt_column", type=str, default=None, help="Column name for prompt/instruction.")
    path_group.add_argument("--completion_column", type=str, default=None, help="Column name for completion/response.")
    
    path_group.add_argument("--model-path", type=str, default=None, help="Path to the model directory.")
    path_group.add_argument("--out-dir", type=str, default="./hierarchos_model", help="Directory to save the new model/adapter.")
    path_group.add_argument("--tokenizer-path", type=str, default=None, help="Path or HF name of the tokenizer.") 
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="Path to a training checkpoint .pt file.")
    path_group.add_argument("--shadow-model-path", type=str, default=None, help="Path to full-precision model for quantized learning.")

    data_fmt_group = parser.add_mutually_exclusive_group()
    data_fmt_group.add_argument("--pre_chunked_dataset", action="store_true")
    data_fmt_group.add_argument("--pre_pt_dataset", action="store_true")

    # --- Architecture Arguments ---
    arch_group = parser.add_argument_group('Architecture')
    arch_group.add_argument("--context_dim", type=int, default=768) 
    arch_group.add_argument("--persistent_dim", type=int, default=128)
    arch_group.add_argument("--ltm_slots", type=int, default=1024)
    arch_group.add_argument("--ltm_key_dim", type=int, default=128)
    arch_group.add_argument("--ltm_val_dim", type=int, default=128)
    arch_group.add_argument("--h_hidden", type=int, default=None)
    arch_group.add_argument("--l_hidden", type=int, default=None)
    arch_group.add_argument("--h_stride", type=int, default=4)
    arch_group.add_argument("--max_h_steps", type=int, default=5)
    arch_group.add_argument("--max_l_steps", type=int, default=5)
    arch_group.add_argument("--ltm_topk", type=int, default=4)
    arch_group.add_argument("--max_length", type=int, default=1024)
    arch_group.add_argument("--auto-max-length", action="store_true")

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--accumulation-steps", "--accumulation_steps", dest="accumulation_steps", type=int, default=1)
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6)
    train_group.add_argument("--disable-lr-schedule", action="store_true")
    train_group.add_argument("--ltm_lr", type=float, default=1e-2)
    train_group.add_argument("--kayla", action="store_true")
    train_group.add_argument("--lora_r", type=int, default=8)
    train_group.add_argument("--lora_alpha", type=int, default=16)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01)
    train_group.add_argument("--commitment-loss-weight", type=float, default=0.5)
    train_group.add_argument("--commitment-threshold", type=float, default=0.05)
    train_group.add_argument("--override-scheduling", action="store_true")
    train_group.add_argument("--save-steps", type=int, default=0, help="Save a checkpoint every N steps (0 to disable).")
    train_group.add_argument("--num_workers", type=int, default=0)
    train_group.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--force-compile", action="store_true")

    # --- Inference & Sampling ---
    infer_group = parser.add_argument_group('Inference')
    infer_group.add_argument("--temperature", type=float, default=0.7)
    infer_group.add_argument("--top-k", type=int, default=40)
    infer_group.add_argument("--top-p", type=float, default=0.9)
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "dml"])
    infer_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))
    
    args = parser.parse_args()

    # Parity: hidden size auto-sync
    if args.mode == 'train' and not args.resume_from_ckpt:
        if args.h_hidden is None: args.h_hidden = args.context_dim
        if args.l_hidden is None: args.l_hidden = args.context_dim

    set_threads(args.threads)
    if args.compile or args.force_compile:
        setup_msvc_environment()
    pt_device = pick_device(args)
    
    # Tokenizer Loading
    tokenizer = None
    tokenizer_source = args.tokenizer_path or args.model_path or "openai-community/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}"); sys.exit(1)

    # --- Auto Max Length Scan ---
    if args.auto_max_length and args.mode in ['train', 'finetune']:
        max_found = 0
        if args.hf_dataset:
            from datasets import load_dataset
            print(f"Scanning HF dataset: {args.hf_dataset}...")
            temp_ds = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
            for sample in tqdm(temp_ds, desc="Scanning HF"):
                processed = process_text_sample(tokenizer, sample, 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column)
                if processed: max_found = max(max_found, len(processed['input_ids']))
        elif args.train and isinstance(args.train, str):
            print(f"Scanning local file: {args.train}...")
            with open(args.train, 'r', encoding='utf-8') as f:
                try: 
                    data = json.load(f)
                    if not isinstance(data, list): data = [data]
                    for obj in tqdm(data, desc="Scanning JSON"):
                        processed = process_text_sample(tokenizer, obj, 9999, args.kayla, prompt_column='instruction', completion_column='output')
                        if processed: max_found = max(max_found, len(processed['input_ids']))
                except:
                    f.seek(0)
                    for line in tqdm(f, desc="Scanning JSONL"):
                        try:
                            processed = process_text_sample(tokenizer, json.loads(line), 9999, args.kayla, prompt_column='instruction', completion_column='output')
                            if processed: max_found = max(max_found, len(processed['input_ids']))
                        except: continue
        if max_found > 0:
            args.max_length = (max_found + 16 + 7) & -8 # Align to 8
            print(f"Auto-scan found max length {max_found}. Setting max_length={args.max_length}")

    # Execution
    if args.mode == "train":
        dataloader = None
        if args.hf_dataset:
            dataset = HuggingFaceMapStyleDataset(temp_ds if 'temp_ds' in locals() else load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split), 
                                                tokenizer, args.max_length, args.kayla, args.text_column, args.prompt_column, args.completion_column)
            dataloader = create_map_style_dataloader(dataset, args.batch_size, tokenizer.pad_token_id, args.num_workers)
        elif args.pre_chunked_dataset:
            dataloader = create_dataloader_for_chunked(args.train, args.max_length, args.batch_size, args.num_workers)
        elif args.pre_pt_dataset:
            dataloader = create_dataloader_pt_chunked(args.train, args.max_length, args.batch_size, args.num_workers)
        elif args.train and isinstance(args.train, str):
            dataset = OriginalJSONLDataset(args.train, tokenizer, args.max_length, args.kayla)
            dataloader = create_map_style_dataloader(dataset, args.batch_size, tokenizer.pad_token_id, args.num_workers)
        
        if dataloader is None:
            print("ERROR: No dataset provided for training. Use --train or --hf_dataset."); sys.exit(1)

        try: dataloader_len = len(dataloader)
        except: dataloader_len = 100000

        train(args, pt_device, tokenizer, dataloader, dataloader_len)
    elif args.mode == "chat":
        chat(args, pt_device, tokenizer)
    else:
        print(f"INFO: Mode '{args.mode}' is not yet fully integrated in the CLI, but modular functions are ready.")

if __name__ == "__main__":
    main()
