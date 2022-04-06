import abc
import re

from lm_eval.base import rf, PerplexityTask
from datasets import load_dataset
from lm_eval.metrics import mean, perplexity, weighted_perplexity, weighted_mean, token_count


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    return tokens


class HumanEvalPerplexity(PerplexityTask, abc.ABC):
    VERSION = 0

    def download(self):
        # Load evaluation dataset and metric
        human_eval = load_dataset("openai_humaneval")

        # Generate completions for evaluation set
        n_tasks = len(human_eval["test"])

        self.prompts = []
        self.solutions = []
        for task in range(n_tasks):
            self.prompts.append(human_eval["test"][task]["prompt"])
            self.solutions.append(human_eval["test"][task]["canonical_solution"])

    def fewshot_description(self):
        # TODO: figure out fewshot description
        return ""

    def has_validation_docs(self):
        return True

    def has_train_docs(self):
        return False

    def has_test_docs(self):
        return False
    
    def validation_docs(self):
        for p, s in zip(self.prompts, self.solutions):
            yield p, s
                
    def train_docs(self):
        pass

    def test_docs(self):
        pass

    def doc_to_target(self, doc):
        return doc[1]
    
    def doc_to_prompt(self, doc):
        return doc[0]
    
    def construct_requests(self, doc, ctx):
        assert not ctx
        req = [rf.loglikelihood_rolling(self.doc_to_prompt(doc)), 
                rf.loglikelihood_rolling(self.doc_to_prompt(doc) + self.doc_to_target(doc))]
        return req

    def process_results(self, doc, results):
        (prompt_loglikelihood, prompt_num_model_tokens), (full_loglikelihood, full_num_model_tokens) = results
        solution_loglikelihood = full_loglikelihood - prompt_loglikelihood
        prompt_words = self.count_words(self.doc_to_prompt(doc))
        solution_words = self.count_words(self.doc_to_target(doc))
        return {
            "prompt_perplexity": (prompt_loglikelihood, prompt_words),
            "solution_perplexity": (solution_loglikelihood, solution_words),
            "full_perplexity": (full_loglikelihood, prompt_words + solution_words),
            "num_prompt_tokens": prompt_words,
            "num_solution_tokens": solution_words,
            "num_full_tokens": prompt_words + solution_words,
            "num_prompt_model_tokens": prompt_num_model_tokens,
            "num_solution_model_tokens": full_num_model_tokens - prompt_num_model_tokens,
            "num_full_model_tokens": full_num_model_tokens,

        }

    def aggregation(self):
        return {
            "prompt_perplexity": weighted_perplexity,
            "solution_perplexity": weighted_perplexity,
            "full_perplexity": weighted_perplexity,
            "num_prompt_tokens": token_count,
            "num_solution_tokens": token_count,
            "num_full_tokens": token_count,
            "num_prompt_model_tokens": token_count,
            "num_solution_model_tokens": token_count,
            "num_full_model_tokens": token_count,
        }
    
    def count_words(self, doc):
        # count number of words in *original doc before detokenization*
        return len(tokenize_for_bleu_eval(doc))

