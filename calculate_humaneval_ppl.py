import json
import math

def ppl(avg_logprob):
    return 2 ** (-avg_logprob / math.log(2))


prompt_log_sum, prompt_model_toks, prompt_canonical_toks = 0, 0, 0
full_log_sum, full_model_toks, full_canonical_toks = 0, 0, 0
solution_log_sum, solution_model_toks, solution_canonical_toks = 0, 0, 0

with open('humaneval_ppl_cushman.jsonl', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        prompt_log_sum += data['prompt']['sum_logprobs']
        prompt_model_toks += len(data['prompt']['tokens'])
        prompt_canonical_toks += len(data['prompt']['canonical_tokens'])

        full_log_sum += data['full']['sum_logprobs']
        full_model_toks += len(data['full']['tokens'])
        full_canonical_toks += len(data['full']['canonical_tokens'])

solution_log_sum = full_log_sum - prompt_log_sum
solution_model_toks = full_model_toks - prompt_model_toks
solution_canonical_toks = full_canonical_toks - prompt_canonical_toks

print('prompt:')
print(ppl(prompt_log_sum / prompt_canonical_toks))
print(prompt_canonical_toks)
print(prompt_model_toks)

print('solution:')
print(ppl(solution_log_sum / solution_canonical_toks))
print(solution_canonical_toks)
print(solution_model_toks)

print('full:')
print(ppl(full_log_sum / full_canonical_toks))
print(full_canonical_toks)
print(full_model_toks)




