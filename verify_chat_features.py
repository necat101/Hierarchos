import sys
import os
import torch
import subprocess
import time

# Define the path to hierarchos.py
HIERARCHOS_PATH = "hierarchos.py"
MODEL_DIR = "./hierarchos_model_test"

def run_chat_test(input_text, args=[]):
    """Runs the chat mode with specific arguments and input."""
    cmd = [sys.executable, HIERARCHOS_PATH, "chat", "--model-path", MODEL_DIR] + args
    
    # We need to pipe input to the subprocess
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd() # Ensure we run from the correct directory
    )
    
    stdout, stderr = process.communicate(input=input_text)
    return stdout, stderr, process.returncode

def create_dummy_model():
    """Creates a dummy model directory for testing if it doesn't exist."""
    if not os.path.exists(MODEL_DIR):
        print(f"Creating dummy model at {MODEL_DIR}...")
        # We can use the 'train' mode to create a tiny model quickly
        # Create a tiny dummy dataset
        with open("dummy_train.jsonl", "w") as f:
            f.write('{"text": "hello world"}\n')
        
        cmd = [
            sys.executable, HIERARCHOS_PATH, "train", 
            "--train", "dummy_train.jsonl",
            "--out-dir", MODEL_DIR,
            "--epochs", "1",
            "--context_dim", "32", # Tiny model
            "--h_hidden", "32",
            "--l_hidden", "32",
            # "--vocab_size", "100", # Removed: Not a valid CLI arg, inferred from tokenizer
            "--max_length", "32",
            "--ltm_slots", "16",
            "--ltm_key_dim", "16",
            "--ltm_val_dim", "16",
            "--disable-lr-schedule"
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Dummy model created.")
        except subprocess.CalledProcessError as e:
            print(f"Error creating dummy model:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
            raise e

def test_sampling_params():
    print("\n--- Testing Sampling Parameters ---")
    
    # Test 1: Temperature setting via slash command
    print("Test 1: Setting Temperature via /temp")
    input_text = "/temp 0.5\n/settings\nexit\n"
    stdout, stderr, _ = run_chat_test(input_text)
    
    if "Set temperature to 0.5" in stdout and "Temperature: 0.5" in stdout:
        print("PASS: Temperature set correctly.")
    else:
        print("FAIL: Temperature setting failed.")
        print(f"FULL STDOUT:\n{stdout}")
        print(f"FULL STDERR:\n{stderr}")

    # Test 2: Top-K setting via slash command
    print("Test 2: Setting Top-K via /topk")
    input_text = "/topk 10\n/settings\nexit\n"
    stdout, stderr, _ = run_chat_test(input_text)
    
    if "Set top_k to 10" in stdout and "Top-K: 10" in stdout:
        print("PASS: Top-K set correctly.")
    else:
        print("FAIL: Top-K setting failed.")
        print(f"FULL STDOUT:\n{stdout}")
        print(f"FULL STDERR:\n{stderr}")

    # Test 3: Top-P setting via slash command
    print("Test 3: Setting Top-P via /topp")
    input_text = "/topp 0.8\n/settings\nexit\n"
    stdout, stderr, _ = run_chat_test(input_text)
    
    if "Set top_p to 0.8" in stdout and "Top-P: 0.8" in stdout:
        print("PASS: Top-P set correctly.")
    else:
        print("FAIL: Top-P setting failed.")
        print(f"FULL STDOUT:\n{stdout}")
        print(f"FULL STDERR:\n{stderr}")

def test_ltm_persistence():
    print("\n--- Testing LTM Persistence ---")
    
    # Test 4: Trigger LTM update and check for save prompt
    print("Test 4: LTM Update Trigger & Save Prompt")
    # We simulate a conversation:
    # 1. User says something (Model generates response)
    # 2. User gives positive feedback (Trigger update on previous turn)
    # 3. User exits
    # 4. User says 'n' to save prompt (to avoid modifying the dummy model permanently, but check if prompt appears)
    
    input_text = "Hello\nGood job\nexit\nn\n" 
    stdout, stderr, _ = run_chat_test(input_text)
    
    if "[Positive feedback. Reinforcing previous memory...]" in stdout:
        print("PASS: Positive feedback triggered LTM update.")
    else:
        print("FAIL: Positive feedback did not trigger update.")
        
    if "Do you want to save the learned LTM updates" in stdout:
        print("PASS: Save prompt appeared on exit.")
    else:
        print("FAIL: Save prompt did not appear.")
        print(f"FULL STDOUT:\n{stdout}")
        print(f"FULL STDERR:\n{stderr}")

if __name__ == "__main__":
    # Ensure we have a model to test with
    try:
        create_dummy_model()
        test_sampling_params()
        test_ltm_persistence()
    except Exception as e:
        print(f"An error occurred: {e}")
