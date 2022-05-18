from datasets import load_dataset

if __name__ == '__main__':
    human_eval = load_dataset("openai_humaneval")

    # Generate completions for evaluation set
    n_tasks = len(human_eval["test"])

    with open('humaneval_to_annotate.txt', 'w') as outfile:
        for task in range(n_tasks):
            outfile.write(human_eval["test"][task]["prompt"] + "====================================================================\n")

