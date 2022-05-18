import argparse
import os
import time
import math
import openai
import json

from tqdm import tqdm
from datasets import load_dataset

# The private OpenAI API key needs to be an environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
# As instructed here: https://community.openai.com/t/token-logprobs-when-echo-is-true/9626/2
# "Transformer models don’t predict the probability of the first token. If you want to get the probability 
# for your first token you can try to use <|endoftext|> as the first token as a workaround."
endoftext_token = '<|endoftext|>'


def oa_completion(**kwargs):
    """ Query OpenAI API for completion.
    Retry with back-off until they respond
    """
    import openai
    backoff_time = 3
    while True:
        try:
            return openai.Completion.create(**kwargs)
        except openai.error.OpenAIError:
            import traceback
            traceback.print_exc()
            time.sleep(backoff_time)
            backoff_time *= 1.5

def call_codex(code_str, temperature):
    eos_code_str = code_str
    # engine: 'davinci-codex' is currently the best codex model
    # max_tokens=0 means that we don't want the model to generate additional tokens
    # logprobs=0 means that we don't want the logprobs of the alternative tokens, only the actual tokens
    # echo=True means that we want the model to echo our prompt, in addition to our (not existing) completion
    completion = oa_completion(engine="code-davinci-001", prompt=eos_code_str,
                                          max_tokens=256,
                                          temperature=temperature,
                                          top_p=0.95,
                                          n=100,
                                          echo=True)
    
    completed_code = [c['text'] for c in completion.choices]

    return completed_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=False, default=os.devnull)
    args = parser.parse_args()
    
    results = {}
    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")

    # Generate completions for evaluation set
    n_tasks = len(human_eval["test"])
    noio_prompts = open('humaneval_only_io.txt').read().split('====================================================================\n')[:-1]
    assert len(noio_prompts) == n_tasks
    solutions = []
    for task in range(n_tasks):
        solutions.append(human_eval["test"][task]["canonical_solution"])


    print('Loaded HumanEval:', len(solutions))

    for temp in [0.0, 0.2, 0.4]:
        with open('code-davinci-001.onlyio.' + str(temp) + '.jsonl', 'w') as out_file:
            for idx, (prompt, solution) in tqdm(enumerate(zip(noio_prompts, solutions))):
                completed_code = call_codex(prompt, temp)
                out_file.write(json.dumps(completed_code) + '\n')
