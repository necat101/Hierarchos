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
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora", "ckpt-2-inf"], help="Operation mode.")

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
    train_group.add_argument("--ltm_lr", type=float, default=1e-3)
    train_group.add_argument("--kayla", action="store_true")
    train_group.add_argument("--lora_r", type=int, default=8)
    train_group.add_argument("--lora_alpha", type=int, default=16)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01)
    train_group.add_argument("--commitment-loss-weight", type=float, default=0.5)
    train_group.add_argument("--commitment-threshold", type=float, default=0.05)
    train_group.add_argument("--l_conv_atol", "--l-conv-atol", type=float, default=1e-4, help="Converge tolerance for WorkerLoop. Default: 1e-4.")
    train_group.add_argument("--detach_every_n_steps", "--detach-every-n-steps", type=int, default=32, help="RWKV state detachment frequency. Default: 32.")
    train_group.add_argument("--h_halt_thresh", "--h-halt-thresh", type=float, default=0.9, help="H-RNN halt probability threshold. Default: 0.9.")
    train_group.add_argument("--encourage-thinking", action="store_true", help="Invert ponder loss to REWARD thinking (for recovery training).")
    train_group.add_argument("--adaptive-ponder", action="store_true", help="Scale ponder target based on CE loss (more thinking for harder content).")
    train_group.add_argument("--ponder-target-scale", type=float, default=0.5, help="Scaling factor for adaptive ponder target. Default: 0.5.")
    train_group.add_argument("--reset-halt-bias", type=float, default=None, metavar="BIAS", help="SURGICAL FIX: Reset h_halt_proj.bias to this value on load (e.g., -2.0 for ~12%% halt prob).")
    train_group.add_argument("--override-scheduling", action="store_true")
    train_group.add_argument("--persist-state", action="store_true", default=False, help="Persist RNN/LTM states between batches. Default: False.")
    train_group.add_argument("--no-persist-state", dest="persist_state", action="store_false", help="Disable state persistence between chunks.")
    train_group.add_argument("--training-chunk-size", "--training_chunk_size", type=int, default=128, help="TBPTT chunk size (Default: 128).")
    train_group.add_argument("--save-steps", type=int, default=0, help="Save a checkpoint every N steps (0 to disable).")
    train_group.add_argument("--num_workers", type=int, default=0)
    train_group.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--force-compile", action="store_true")

    # --- Evaluation Arguments (lm-evaluation-harness) ---
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument("--eval-tasks", type=str, nargs='+', default=None,
        help="Benchmark tasks to run during training (e.g., 'hellaswag arc_easy'). Disabled by default. Requires: pip install lm-eval")
    eval_group.add_argument("--eval-every-epoch", type=int, default=1,
        help="Run evaluation every N epochs (default: 1).")
    eval_group.add_argument("--eval-batch-size", type=int, default=1,
        help="Batch size for evaluation (default: 1).")
    eval_group.add_argument("--eval-limit", type=int, default=None,
        help="Limit samples per task for fast evaluation runs (e.g., 10 for quick tests).")
    eval_group.add_argument("--eval-steps", type=int, default=None,
        help="Run evaluation every N training steps (for quick testing). Triggers periodically.")

    # --- Inference & Sampling ---
    infer_group = parser.add_argument_group('Inference')
    infer_group.add_argument("--temperature", type=float, default=0.7)
    infer_group.add_argument("--top-k", type=int, default=40)
    infer_group.add_argument("--top-p", type=float, default=0.9)
    infer_group.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty for repeating tokens (1.0=none, >1.0=discourage). Default: 1.2.")
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "dml"])
    infer_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))
    
    # --- Chat-specific (Online Learning) ---
    chat_group = parser.add_argument_group('Chat Online Learning')
    chat_group.add_argument("--enable-quantized-learning", action="store_true", help="Enable LTM updates for quantized models.")
    chat_group.add_argument("--ltm-lora-path", type=str, default=None, help="Path to save/load LTM updates as delta file.")
    chat_group.add_argument("--static-ltm-lr", action="store_true", help="Disable cosine annealing for LTM updates.")
    chat_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="Cosine cycle steps for LTM learning.")
    chat_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="Min LR for LTM cosine annealing.")
    chat_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="Target %% of params to train (overrides lora_r).")
    chat_group.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    chat_group.add_argument("--passive-learning", action="store_true", default=True, help="Enable passive LTM learning after each generation turn (ON by default).")
    chat_group.add_argument("--no-passive-learning", dest="passive_learning", action="store_false", help="Disable passive LTM learning.")
    chat_group.add_argument("--passive-lr", type=float, default=5e-6, help="Learning rate for passive LTM updates (default: 5e-6, very conservative).")
    chat_group.add_argument("--surprise-threshold", type=float, default=1.0, help="Only learn when loss > threshold (default: 1.0 for conservative learning).")
    
    # --- Utility Arguments ---
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument("--ckpt-input", type=str, default=None, help="Input checkpoint path for ckpt-2-inf mode.")
    util_group.add_argument("--inf-output", type=str, default=None, help="Output inference model path for ckpt-2-inf mode.")
    util_group.add_argument("--ckpt-tok-path", type=str, default=None, help="HuggingFace tokenizer name/path to embed in the inference model (e.g., 'gpt2', 'openai-community/gpt2').")
    
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
    elif args.mode == "finetune":
        # Build dataloader for finetune
        dataloader = None
        if args.hf_dataset:
            from datasets import load_dataset
            temp_ds = load_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
            dataset = HuggingFaceMapStyleDataset(temp_ds, tokenizer, args.max_length, args.kayla, 
                                                  args.text_column, args.prompt_column, args.completion_column)
            dataloader = create_map_style_dataloader(dataset, args.batch_size, tokenizer.pad_token_id, args.num_workers)
        elif args.pre_chunked_dataset:
            dataloader = create_dataloader_for_chunked(args.train, args.max_length, args.batch_size, args.num_workers)
        elif args.pre_pt_dataset:
            dataloader = create_dataloader_pt_chunked(args.train, args.max_length, args.batch_size, args.num_workers)
        elif args.train and isinstance(args.train, str):
            dataset = OriginalJSONLDataset(args.train, tokenizer, args.max_length, args.kayla)
            dataloader = create_map_style_dataloader(dataset, args.batch_size, tokenizer.pad_token_id, args.num_workers)
        
        if dataloader is None:
            print("ERROR: No dataset provided for finetuning. Use --train or --hf_dataset."); sys.exit(1)

        try: dataloader_len = len(dataloader)
        except: dataloader_len = 100000

        finetune(args, pt_device, tokenizer, dataloader, dataloader_len)
    elif args.mode == "ckpt-2-inf":
        # Convert checkpoint to inference model (HuggingFace-style directory)
        ckpt_path = args.ckpt_input or args.resume_from_ckpt or args.model_path
        if not ckpt_path:
            print("ERROR: No checkpoint specified. Use --ckpt-input, --resume-from-ckpt, or --model-path.")
            sys.exit(1)
        
        # Determine output directory (strip .pt extension if provided)
        if args.inf_output:
            output_dir = args.inf_output.replace('.pt', '')
        else:
            base_dir = os.path.dirname(ckpt_path)
            output_dir = os.path.join(base_dir, "hierarchos_final")
        
        print(f"Converting checkpoint to inference model...")
        print(f"  Input:  {ckpt_path}")
        print(f"  Output: {output_dir}/")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # Extract and clean state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {}
        for k, v in state_dict.items():
            # Remove _orig_mod. prefix from compiled models
            clean_key = k.replace('_orig_mod.', '')
            clean_state_dict[clean_key] = v
        
        # Extract config
        config = checkpoint.get('config', {})
        completed_epoch = checkpoint.get('completed_epoch', checkpoint.get('epoch', 'unknown'))
        
        # Handle tokenizer
        tokenizer_name = args.ckpt_tok_path or config.get('tokenizer_name', 'openai-community/gpt2')
        print(f"  Tokenizer: {tokenizer_name}")
        
        # Load and verify tokenizer
        try:
            inf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            vocab_size = len(inf_tokenizer)
            print(f"  Vocab size: {vocab_size}")
            
            # Verify vocab size matches model
            model_vocab = config.get('vocab_size')
            if model_vocab and model_vocab != vocab_size:
                print(f"  WARNING: Model vocab_size ({model_vocab}) != tokenizer vocab_size ({vocab_size})")
                print(f"           Make sure you're using the same tokenizer as training!")
        except Exception as e:
            print(f"ERROR: Failed to load tokenizer '{tokenizer_name}': {e}")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer files
        print(f"  Saving tokenizer files...")
        inf_tokenizer.save_pretrained(output_dir)
        
        # Create inference-ready checkpoint
        model_path = os.path.join(output_dir, "model.pt")
        inference_checkpoint = {
            'model_state_dict': clean_state_dict,
            'config': config,
            'completed_epoch': completed_epoch,
            'training_complete': True,
            'converted_from': os.path.basename(ckpt_path),
            'tokenizer_name': tokenizer_name,
        }
        torch.save(inference_checkpoint, model_path)
        
        # Save config as JSON for easy inspection
        config_path = os.path.join(output_dir, "hierarchos_config.json")
        import json as json_module
        config_to_save = dict(config)
        config_to_save['completed_epoch'] = completed_epoch
        config_to_save['tokenizer_name'] = tokenizer_name
        config_to_save['converted_from'] = os.path.basename(ckpt_path)
        with open(config_path, 'w') as f:
            json_module.dump(config_to_save, f, indent=2, default=str)
        
        # Report
        input_size = os.path.getsize(ckpt_path)
        output_size = os.path.getsize(model_path)
        reduction = (1 - output_size / input_size) * 100
        
        # Count all files in output dir
        total_output_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir))
        
        print(f"\n" + "="*60)
        print(f"CONVERSION COMPLETE!")
        print(f"="*60)
        print(f"Input checkpoint: {input_size / 1e6:.2f} MB")
        print(f"Output directory: {output_dir}/")
        print(f"  - model.pt:     {output_size / 1e6:.2f} MB  ({reduction:.1f}% reduction)")
        print(f"  - Total size:   {total_output_size / 1e6:.2f} MB")
        print(f"  - Epoch:        {completed_epoch}")
        print(f"  - Tokenizer:    {tokenizer_name}")
        print(f"\nTo use the model for inference:")
        print(f"  python hierarchos_cli.py chat --model-path \"{output_dir}\"")
        print(f"="*60)
    else:
        print(f"INFO: Mode '{args.mode}' is not yet fully integrated in the CLI.")

if __name__ == "__main__":
    main()
