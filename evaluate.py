import subprocess
import json
import os

RESULTS_DIR = "output"

MODELS = [
    "huggyllama/llama-30b",
    "huggyllama/llama-13b",
    "huggyllama/llama-7b",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
    "facebook/opt-66b",
    "facebook/opt-350m",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
    "facebook/opt-2.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
]

TASKS = ",".join(['wikitext', 'penn_treebank', 'lambada_standard', 'mnli'])


if __name__ == '__main__':
    for model in MODELS:
        model_fname = model.replace('/', '_')
        try:
            subprocess.run(['python3', 'main.py', '--model', 'hf-causal-experimental', '--model_args',
                            f'pretrained={model},use_accelerate=True', '--tasks', TASKS, '--batch_size',
                            '1', '--write_out', '--output_path', f'{RESULTS_DIR}/{model_fname}.json']
                            check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            with open(f'{RESULTS_DIR}/errors.log', 'a') as f:
                f.write(f'=== {model} ===')
                f.write(e.stdout)
                f.write(e.stderr)
    # combine results into single file
    results = []
    for output_fname in os.listdir(RESULTS_DIR):
        if output_fname != "errors.log":
            with open(f'{RESULTS_DIR}/{output_fname}') as f:
                results.append(json.load(f))
    with open(f'{RESULTS_DIR}/all_results.json', 'w') as f:
        json.dump(results, f, indent=4)