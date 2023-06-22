import subprocess
import json
import os
from datetime import datetime

RESULTS_DIR = "output"


MODELS = [
    "EleutherAI/gpt-j-6b",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125m",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/gpt-neox-20b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-70m",
    "THUDM/glm-10b",
    "THUDM/glm-2b",
    "bigscience/bloom",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-560m",
    "bigscience/bloom-7b1",
    "decapoda-research/llama-13b-hf",
    "decapoda-research/llama-30b-hf",
    "decapoda-research/llama-65b-hf",
    "decapoda-research/llama-7b-hf",
    "facebook/opt-1.3b",
    "facebook/opt-125m",
    "facebook/opt-13b",
    "facebook/opt-2.7b",
    "facebook/opt-30b",
    "facebook/opt-350m",
    "facebook/opt-6.7b",
    "facebook/opt-66b",
    "huggyllama/llama-13b",
    "huggyllama/llama-30b",
    "huggyllama/llama-7b",
]

TASKS = ",".join(['wikitext', 'penn_treebank', 'lambada_standard', 'mnli'])


if __name__ == '__main__':
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    for model in MODELS:
        print(f"{datetime.now().isoformat()}: running {model} on {TASKS}")
        model_fname = model.replace('/', '_')
        try:
            subprocess.run(['python3', 'main.py', '--model', 'hf-causal-experimental',
                            '--model_args', f'pretrained={model},use_accelerate=True', '--tasks',
                            TASKS, '--batch_size', 'auto', '--write_out', '--output_path',
                            f'{RESULTS_DIR}/{model_fname}.json'], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"{datetime.now().isoformat()}: ERROR {model} on {TASKS}")
            with open(f'{RESULTS_DIR}/errors.log', 'a') as f:
                f.write(f'=== {model} ===')
                f.write(str(e.stdout))
                f.write(str(e.stderr))
    # combine results into single file
    results = []
    for output_fname in os.listdir(RESULTS_DIR):
        if output_fname != "errors.log":
            with open(f'{RESULTS_DIR}/{output_fname}') as f:
                results.append(json.load(f))
    with open(f'{RESULTS_DIR}/all_results.json', 'w') as f:
        json.dump(results, f, indent=4)