import argparse
import os
import time
import math
import openai
import json

from tqdm import tqdm
from datasets import load_dataset
from lm_eval.tasks.humaneval_ppl import tokenize_for_bleu_eval

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

def call_codex(code_str):
    eos_code_str = endoftext_token + code_str
    # engine: 'davinci-codex' is currently the best codex model
    # max_tokens=0 means that we don't want the model to generate additional tokens
    # logprobs=0 means that we don't want the logprobs of the alternative tokens, only the actual tokens
    # echo=True means that we want the model to echo our prompt, in addition to our (not existing) completion
    completion = oa_completion(engine="cushman-codex", prompt=eos_code_str,
                                          max_tokens=0,
                                          temperature=0.0,
                                          logprobs=0,
                                          n=1,
                                          echo=True)
    
    c = completion.choices[0]
    # skipping the <|endoftext|> token
    saved_probs = {
        'text': code_str,
        'canonical_tokens': tokenize_for_bleu_eval(code_str),
        'tokens': c.logprobs.tokens[1:],
        'logprobs': c.logprobs.token_logprobs[1:],
        'sum_logprobs': sum(c.logprobs.token_logprobs[1:])
    }

    return saved_probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=False, default=os.devnull)
    args = parser.parse_args()
    
    results = {}
    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")

    # Generate completions for evaluation set
    n_tasks = len(human_eval["test"])

    prompts = []
    solutions = []
    for task in range(n_tasks):
        prompts.append(human_eval["test"][task]["prompt"])
        solutions.append(human_eval["test"][task]["canonical_solution"])

    print('Loaded HumanEval:', len(prompts))
    
    log_probs_sum = 0
    tokens_count = 0
    ignored_files = []
    all_per_token_probs = []
    with open(args.output, 'w') as out_file:
        for idx, (prompt, solution) in tqdm(enumerate(zip(prompts, solutions))):
            prompt_per_token_probs = call_codex(prompt)
            full_per_token_probs = call_codex(prompt + solution)
            data_dict = {'prompt': prompt_per_token_probs, 'full': full_per_token_probs}
            out_file.write(json.dumps(data_dict) + '\n')
