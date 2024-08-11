# This template file is adapted from: https://github.com/EleutherAI/lm-evaluation-harness/blob/master/templates/new_task.py

# TODO: Remove all TODO comments once the implementation is complete.
"""
TODO: Add the Paper Title on this line.
TODO: Add the paper's PDF URL (preferably from arXiv) on this line.
TODO: Write a Short Description of the task.
Homepage: TODO: Add the URL to the task's Homepage here.
"""
from bigcode_eval.base import Task
from evaluate import load
import json

# TODO: Add the BibTeX citation for the task.
_CITATION = """
@misc{muennighoff2024octopackinstructiontuningcode,
      title={OctoPack: Instruction Tuning Code Large Language Models}, 
      author={Niklas Muennighoff and Qian Liu and Armel Zebaze and Qinkai Zheng and Binyuan Hui and Terry Yue Zhuo and Swayam Singh and Xiangru Tang and Leandro von Werra and Shayne Longpre},
      year={2024},
      eprint={2308.07124},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.07124}, 
}
"""


# TODO: Replace `NewTask` with the name of your Task.
class CompareEval(Task):
    # TODO: Add the `DATASET_PATH` string. This will be the name of the `Task`
    # dataset as denoted in HuggingFace `datasets`.
    DATASET_PATH = "Sam137/CompareEval"
    # TODO: Add the `DATASET_NAME` string. This is the name of a subset within
    # `DATASET_PATH`. If there aren't specific subsets you need, leave this as `None`.
    DATASET_NAME = None

    def __init__(self):
        super().__init__(
            # TODO: Specify the list of stop words in `stop_words` for the code generation task \
            # and if the evaluation requires executing the generated code in `requires_execution`.
            stop_words=["<|EOS|>"],
            requires_execution=False,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return self.dataset['train']

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/compareeval.json", "r"
        ) as file:
            examples = json.load(file)
        return examples
    
    def get_instruction(self, old_code, new_code):
        return f"Summarize the update between the two Python code using one sentence only in the form of a commit message. First code version:\n{old_code}\nSecond code version:\n{new_code}\n"

    def get_prompt(self, doc):
        """
        Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        inst = f"Summarize the update between the two Python code using one sentence only in the form of a commit message. First code version:\n{doc['old_contents']}\nSecond code version:\n{doc['new_contents']}\n"
        examples = self.fewshot_examples()
        few_shot_prompt = f"### Instruction:\n{self.get_instruction(examples['old_code'], examples['new_code'])}\n### Response:\n{examples['response']}"
        prompt = f"You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer\n{few_shot_prompt}\n### Instruction:\n{inst}\n### Response:\n"
        return prompt

    def get_reference(self, doc):
        # TODO: get the reference solution from a sample `doc` from the dataset
        """
        Builds the reference solution for the doc (sample from the test dataset).
        :param doc: dict[str: str]
            sample from the test dataset
        :return: str
        """
        return doc['subject'].strip()

    def postprocess_generation(self, generation, idx):
        # TODO: define the postprocessing for the LM generation
        """
        Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int (if needed)
            index of doc in the dataset to which the generation belongs
        :return: str
        """
        prompt = self.get_prompt(self.get_dataset()[idx])
        generation = generation[len(prompt):].strip()
        for word in self.stop_words:
            if word in generation:
                generation = generation[:generation.find(word)]
        return generation.rstrip()

    def process_results(self, generations, references):
        # TODO: define how the evaluation score is computed from list of \
        # generations and reference solutions
        """
        Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        We encourage to directly load the metric from `evaluate` library to keep the code concise.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        bleu = load("bleu")
        gens = [gen[0] for gen in generations]
        results = bleu.compute(
            references=references, predictions=gens)
        return results

if __name__ == "__main__":
    #just testing a bit
    task = CompareEval()
    ds = task.get_dataset()
    for sample in ds:
        print(task.get_prompt(sample))
        break