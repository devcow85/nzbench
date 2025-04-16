import json
import re

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np
import pandas as pd

class BenchmarkQAataLoader:
    def __init__(self, prompt_generator, n_shots = 0):
        self.prompt_generator = prompt_generator
        self.n_shots = n_shots
    
    def get_prefix_prompts(self, task='all'):
        return_dict = {}
            
        for subject in (self.task if task == 'all' else [task]):
            prompt = self.prefix_prompt.format(subject)
            
            ref_key = "category" if "category" in self.ref_data.features.keys() else "subject"
            if self.n_shots > 0:
                start_ref_index = self.ref_data[ref_key].index(subject)
                for k in range(self.n_shots):
                    prompt += self.prompt_generator(self.ref_data[start_ref_index+k], self.choices, last=False)
            return_dict[subject] = prompt
        return return_dict
    
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        # get info
        sub_key = "category" if "category" in self.test_data.features.keys() else "subject"
        
        subject = self.test_data[idx][sub_key]
        answer = self.test_data[idx]['answer']
        
        # gen prompt
        prompt = ""
        if self.enable_prefix_promts:
            prompt = self.prefix_prompt.format(subject)
        
        # add n-shot qa from ref_data
        if self.n_shots > 0:
            start_ref_index = self.ref_data[sub_key].index(subject)
            for k in range(self.n_shots):
                prompt += self.prompt_generator(self.ref_data[start_ref_index+k], self.choices, last=False)
        
        # add current qa
        prompt += self.prompt_generator(self.test_data[idx], self.choices)
        
        return prompt, subject, answer

def mmlupro_prompt_generator(data, choices, last = True):
    prompt = "Question:\n"
    prompt += data['question'] + "\n"
    prompt += "Options:\n"

    for i in range(len(data['options'])):
        prompt += "{}. {}\n".format(choices[i], data['options'][i])
    if last:
        prompt += "Answer: Let's think step by step."
    else:
        cot_content = data["cot_content"].replace("A: Let's think step by step.",
                                                     "Answer: Let's think step by step.")
        prompt += cot_content + "\n\n"
        
    return prompt

def extract_choice(text):
    return text[-1]

def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        print("1st answer extract failed\n" + text)
        return extract_again(text)

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None
    
def mmlu_prompt_generator(data, choices, last = True):
    prompt = data['question']
    for i in range(len(choices)):
        prompt += "\n{}. {}".format(choices[i], data['choices'][i])
    prompt += "\nAnswer:"
    if not last:
        prompt += " {}\n\n".format(choices[data['answer']])
        
    return prompt

class NZBenchmark:
    def __init__(self, benchmark, model, device="cuda:0", output_fd="output.json"):
        self.benchmark = benchmark
        self.model = model
        self.device = device
        self.output_fd = output_fd

        # load mode and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # load dataset
        if benchmark == "mmlu":
            self.dataset = MMLUDataLoader()
            self.generate_params = {
                    "do_sample": True,
                    "max_new_tokens": 1,
                    "top_k": 1,
                    "temperature": 1.0,
                    "use_cache":True,
                    "pad_token_id":self.tokenizer.pad_token_id,  # generate에 명시 전달

                }
            self.extract_fn = extract_choice
            
        elif benchmark == "mmlu_pro":
            self.dataset = MMLUProDataLoader()
            self.generate_params = {
                    "do_sample": True,
                    "max_new_tokens": 1024,
                    "top_k": 1,
                    "temperature": 1.0,
                    "use_cache":True,
                    "pad_token_id":self.tokenizer.pad_token_id,  # generate에 명시 전달
                }
            self.extract_fn = extract_answer
        else:
            raise ValueError("Invalid benchmark name")
        
        self.model.to(self.device)  # device
        self.model.eval()    # eval mode
        
    def _save_log(self, log):
        with open(self.output_fd, 'w') as f:
            json.dump(log, f, indent=2, ensure_ascii=False)
    
    def run(self, kv_cache_reuse=False):
        pbar = tqdm(self.dataset, desc=self.dataset.__class__.__name__)
        
        log = []
        acc=0
        total=0
        
        for idx, (question, subject, answer) in enumerate(pbar):
            inputs = self.tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(self.device)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            out = self.model.generate(**inputs, **self.generate_params)
            
            pred = self.tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            pred_answer = self.extract_fn(pred)
            
            if isinstance(answer, int):
                answer = self.dataset.choices[answer]
            
            log_dict = {"question": question,
                    "response": pred_answer,
                    "answer": answer,
                    "subject": subject}
            log.append(log_dict)
            
            self._save_log(log)
    
            acc += 1 if pred_answer.strip() == answer else 0
            total += 1
        
            pbar.set_postfix({
                        "subject": subject,
                        "accuracy": f"{acc/total:.3f}"
                    })
    
def benchmark_summary(log):
    df = pd.DataFrame(log)
    
    df["correct"] = df.apply(lambda row: row["response"][0] == ["A", "B", "C", "D"][row["answer"]], axis=1)

    subject_accuracy = df.groupby("subject")["correct"].mean().reset_index()
    subject_accuracy.columns = ["subject", "accuracy"]
    
    overall_accuracy = df["correct"].mean()
    total_execution_time = df["e2e"].sum()

    execution_time_stats = df.groupby("subject").agg(
    mean_e2e=("e2e", np.nanmean),
    median_e2e=("e2e", np.nanmedian),
    p50_e2e=("p50", np.nanmean),
    p99_e2e=("p99", np.nanmean)).reset_index()
    
    print("### Subject Accuracy ###")
    print(subject_accuracy)

    print("\n### Execution Time Statistics ###")
    print(execution_time_stats)

    print("\n### Overall Accuracy ###")
    print(f"Overall Accuracy: {overall_accuracy:.2%}")  
    print(f"Total Execution Time (e2e): {total_execution_time/1000:.2f} s")
    
class MMLUDataLoader(BenchmarkQAataLoader):
    prefix_prompt = 'The following are multiple choice questions (with answers) about {}.\n\n'
    
    def __init__(self, dataset = "hails/mmlu_no_train", prompt_generator=mmlu_prompt_generator, enable_prefix_promts = True, n_shots = 0):
        self.test_data = load_dataset(dataset, 'all', trust_remote_code = True)['test']
        self.ref_data = load_dataset(dataset, 'all', trust_remote_code = True)['dev']

        self.choices = self.test_data.features['answer']._int2str
        self.enable_prefix_promts = enable_prefix_promts
        self.task = np.unique(self.ref_data['subject'])
        super().__init__(prompt_generator, n_shots)
    
class MMLUProDataLoader(BenchmarkQAataLoader):
    prefix_prompt = 'The following are multiple choice questions (with answers) about {}. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.\n'
    
    def __init__(self, dataset = "TIGER-Lab/MMLU-Pro", prompt_generator=mmlupro_prompt_generator, enable_prefix_promts = True, n_shots = 0):
        self.test_data = load_dataset(dataset)['test']
        self.ref_data = load_dataset(dataset)['validation']
        
        self.choices = [chr(65 + i) for i in range(10)] # A~J
        self.enable_prefix_promts = enable_prefix_promts
        self.task = np.unique(self.ref_data['category'])
        super().__init__(prompt_generator, n_shots)
        
