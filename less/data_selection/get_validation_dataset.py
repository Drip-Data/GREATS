import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

# llama-chat model's instruction format
B_INST, E_INST = "[INST]", "[/INST]"


def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             max_res_length: int,
             print_ex: bool = False) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer used to encode the input.
        query (str): The question part of the chat conversation.
        completion (str): The answer part of the chat conversation.
        max_length (int): The maximum length of the input tensors.
        print_ex (bool, optional): Whether to print the example. Defaults to False.

    Returns:
        tuple: A tuple containing the full input IDs, labels, and attention mask tensors.
    """
    full_prompt = query + completion

    if print_ex:
        print("******** Example starts ********")
        print(full_prompt)
        print("******** Example ends ********")


    ######### 这样可能会导致comletion被截断 #########
    prompt_input_ids = torch.tensor(
        tokenizer.encode(query, max_length=max_length))
    full_input_ids = torch.tensor(
        tokenizer.encode(full_prompt, max_length=max_length))
    labels = torch.tensor(tokenizer.encode(full_prompt, max_length=max_length))
    labels[:len(prompt_input_ids)] = -100
    attention_mask = [1] * len(full_input_ids)
    # tokenizer.padding_side = 'left'
    # prompt_input = tokenizer(query, max_length=max_length, return_tensors='pt', truncation=True, padding='max_length')
    # tokenizer.padding_side = 'right'
    # comletion_input = tokenizer(completion, max_length=max_res_length, return_tensors='pt', truncation=True, padding='max_length')
    # full_input_ids = torch.cat([prompt_input['input_ids'], comletion_input['input_ids']], dim=1)
    # attention_mask = torch.cat([prompt_input['attention_mask'], comletion_input['attention_mask']], dim=1)
    
    # labels = full_input_ids.clone()
    # labels[:, :prompt_input['input_ids'].shape[1]] = -100
    # labels[:, prompt_input['input_ids'].shape[1]:][comletion_input['attention_mask']==0] = -100

    return full_input_ids, labels, attention_mask


######## 数据处理的函数需要重写，利用dataset.map
def get_bbh_dataset(data_dir: str,
                    tokenizer: PreTrainedTokenizerBase,
                    max_length: int,
                    max_res_length: int,
                    use_chat_format: bool = True,
                    chat_format: str = "tulu",
                    validation=False,
                    k = 3,
                    **kwargs):
    """
    Get the bbh dataset in the instruction tuning format. Each example is formatted as follows: 

    Query: 
    <|user|>
    <Task Prompt>
    <Ex1>
    <Ex2>
    <Question of Ex3>
    <|assistant|>
    A:

    Completion:
    <Answer of Ex3>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the input. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        n_shot (int, optional): The number of shots for few-shot learning. Defaults to 3 for bbh.

    Returns:
        Dataset: The BBH dataset containing input_ids, attention_mask, and labels.
    """
    file = f"{data_dir}/eval/bbh/bbh-three-shot.json"

    bbh_few_shot_examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}

    # there are multiple tasks in the bbh dataset
    # each task has 3 examples
    for task in bbh_few_shot_examples:
        few_shot_exs = bbh_few_shot_examples[task]

        stuff = few_shot_exs.split("\n\n")
        exes = stuff[-3:]
        
        task_prompt = "\n\n".join(stuff[:-3])

        def form_icl(exs):
            string = ""
            for ex in exs:
                question, answer = ex.split("\nA:")
                string += question + "\nA:" + answer
                string += "\n\n"
            return string
        
        for i in range(k):
            target_ex = exes[i]
            other_exes = exes[:i] + exes[i+1:]
            icl = form_icl(other_exes)
            question, answer = target_ex.split("\nA:")

            if use_chat_format:
                if chat_format == "tulu":  # we follow the tulu instruction tuning format
                    question = "<|user|>\n" + task_prompt.strip() + "\n\n" + icl + \
                        f"{question}" + "\n<|assistant|>\nA:"
                else:
                    question = f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
            else:
                question = task_prompt.strip() + "\n\n" + \
                    f"{question}" + "\nA:"
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, question, answer, max_length, max_res_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset

def get_tydiqa_dataset_df(data_dir: str,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       validation=False,
                       k = 5):
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。", "文章:", "问题:", "答案:")

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-goldp-v1.1-dev.json"
    # grad-tracin/LESS/data/eval/tydiqa/dev/tydiqa-goldp-v1.1-dev.json
    if validation:
        file = os.path.join(f"{data_dir}/eval/tydiqa/dev", file_name)
    else:
        file = os.path.join(f"{data_dir}/eval/tydiqa/test", file_name)

    examples = json.load(open(file, "r"))
    # shuffle(examples["data"])
    import random
    random.seed(42)
    examples_data = examples["data"]
    random.shuffle(examples_data)

    dataset = []

    for i, example in enumerate(examples_data):
        if i == k:
            break
        ID = example["paragraphs"][0]["qas"][0]['id']
        lang = ID.split("-")[0]

        context = example["paragraphs"][0]["context"]
        question = example["paragraphs"][0]["qas"][0]["question"]
        answer = example["paragraphs"][0]["qas"][0]["answers"][0]["text"]

        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += p_template + " " + \
            context + "\n" + q_template + \
            " " + question + "\n"
        answer = " " + answer
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        if validation:
            print("########## Example {} ##########".format(i))
            print(prompt)
            print(answer)
            dataset.append((prompt, answer, lang))
        else:
            dataset.append((prompt, answer, lang))

    return dataset


def get_tydiqa_dataset(data_dir: str,
                       tokenizer: PreTrainedTokenizerBase,
                       max_length: int,
                       max_res_length: int,
                       use_chat_format: bool = True,
                       chat_format: str = "tulu",
                       zh: bool = False,
                       validation=False,
                       k = 5,
                       **kwargs) -> Dataset:
    """
    Get the tydiqa dataset in the instruction tuning format. Each example is formatted as follows:  

    Query: 
    <|user|>
    <Task Prompt>
    <Passage>
    <Question>
    <|assistant|>
    Answer: 

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use for tokenization.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format. Defaults to True.
        chat_format (str, optional): The chat format to use. Defaults to "tulu".
        zh (bool, optional): Whether to use the Chinese validation examples. Defaults to False.


    Returns:
        Dataset: The tokenized TydiQA dataset.
    """

    # Same template as https://github.com/allenai/open-instruct/blob/main/eval/tydiqa/run_eval.py#L17
    encoding_templates_with_context = {
        "english": ("Answer the following question based on the information in the given passage.", "Passage:", "Question:", "Answer:"),
        "arabic": ("أجب على السؤال التالي بناءً على المعلومات في المقطع المعطى.", "المقطع:", "السؤال:", "الإجابة:"),
        "bengali": ("প্রদত্ত অধ্যায়ের তথ্যের উপর ভিত্তি করে নিম্নলিখিত প্রশ্নের উত্তর দিন।", "অধ্যায়:", "প্রশ্ন:", "উত্তর:"),
        "finnish": ("Vastaa seuraavaan kysymykseen annetun kappaleen tiedon perusteella.", "Kappale:", "Kysymys:", "Vastaus:"),
        "indonesian": ("Jawab pertanyaan berikut berdasarkan informasi di bagian yang diberikan.", "Bagian:", "Pertanyaan:", "Jawaban:"),
        "korean": ("주어진 문단의 정보에 기반하여 다음 질문에 답하십시오.", "문단:", "질문:", "답변:"),
        "russian": ("Ответьте на следующий вопрос на основе информации в данном отрывке.", "Отрывок:", "Вопрос:", "Ответ:"),
        "swahili": ("Jibu swali lifuatalo kulingana na habari kwenye kifungu kilichotolewa.", "Kifungu:", "Swali:", "Jibu:"),
        "telugu": ("ఇచ్చిన పేరాలోని సమాచారం ఆధారంగా కింది ప్రశ్నకు సమాధానం ఇవ్వండి.", "పేరా:", "ప్రశ్న:", "సమాధానం:")
    }

    # Chinese validation examples
    if zh:
        for lang in encoding_templates_with_context:
            encoding_templates_with_context[lang] = (
                "根据所给文章中的信息回答以下问题。", "文章:", "问题:", "答案:")

    file_name = "tydiqa-one-shot-zh.json" if zh else "tydiqa-goldp-v1.1-dev.json"
    # grad-tracin/LESS/data/eval/tydiqa/dev/tydiqa-goldp-v1.1-dev.json
    if validation:
        file = os.path.join(f"{data_dir}/eval/tydiqa/dev", file_name)
    else:
        file = os.path.join(f"{data_dir}/eval/tydiqa/test", file_name)

    examples = json.load(open(file, "r"))
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    
    import random
    random.seed(42)

    
    examples_data = examples["data"]
    random.shuffle(examples_data)




    for i, example in enumerate(examples_data):
        if i == k:
            break
        ID = example["paragraphs"][0]["qas"][0]['id']
        lang = ID.split("-")[0]

        context = example["paragraphs"][0]["context"]
        question = example["paragraphs"][0]["qas"][0]["question"]
        answer = example["paragraphs"][0]["qas"][0]["answers"][0]["text"]

        prompt, p_template, q_template, a_template = encoding_templates_with_context[lang]
        prompt += p_template + " " + \
            context + "\n" + q_template + \
            " " + question + "\n"
        answer = " " + answer
        if use_chat_format:
            if chat_format == "tulu":
                prompt = "<|user|>\n" + prompt + "<|assistant|>\n" + a_template
            else:
                prompt = f"<s> {B_INST} {prompt} {E_INST} {a_template}"
        else:
            prompt = prompt + a_template
        if validation:
            print("########## Example {} ##########".format(i))
            print(prompt)
            print(answer)
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, max_res_length, print_ex=False)
            print(len(full_input_ids))
            
            print("################################")
        else:
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, max_res_length, print_ex=False)
        dataset["input_ids"].append(full_input_ids)
        dataset["labels"].append(labels)
        dataset["attention_mask"].append(attention_mask)

    dataset = Dataset.from_dict(dataset)
    return dataset



def get_mmlu_dataset_df(data_dir: str,
                        validation=False,
                        k = 5,
                        subject = 'abstract_algebra'):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    subjects = [subject]
    for subject in subjects:
        if validation:
            df = pd.read_csv(os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None)[:k]
            df = df[:min(k, len(df))]
        else:
            df = pd.read_csv(os.path.join(mmlu_data_dir, "test", subject + "_test.csv"), header=None)
            df = df[:min(k, len(df))]
    return df



def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     max_res_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     validation=False,
                     k = 5,
                     subject = 'abstract_algebra', 
                     **kwargs):
    """
    Get the MMLU dataset in the instruction tuning format. Each example is formatted as follows:

    Query:
    <|user|>
    <Task Prompt>
    <Question>
    <|assistant|>
    The answer is:

    Completion:
    <Answer>

    Args:
        data_dir (str): The main data directory.
        tokenizer (Tokenizer): The tokenizer used to tokenize the input text.
        max_length (int): The maximum length of the input sequence.
        use_chat_format (bool, optional): Whether to use chat format for the prompts. Defaults to True.
        chat_format (str, optional): The chat format to use for the prompts. Defaults to "tulu".

    Returns:
        Dataset: The tokenized dataset containing input_ids, attention_mask, and labels.
    """
    mmlu_data_dir = os.path.join(data_dir, "eval", "mmlu")
    # subjects = sorted(
    #     [
    #         f.split("_test.csv")[0]
    #         for f in os.listdir(os.path.join(mmlu_data_dir, "test"))
    #         if "_test.csv" in f
    #     ]
    # )
    subjects = [subject]

    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s

    def gen_prompt(train_df, subject, i=0):
        prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        prompt += format_example(train_df, i, include_answer=False)
        return prompt

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2
        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"
        return prompt

    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    for subject in subjects:

        if validation:
            dev_df = pd.read_csv(os.path.join(mmlu_data_dir, "dev", subject + "_dev.csv"), header=None)[:k]
            dev_df = dev_df[:min(k, len(dev_df))]
        else:
            dev_df = pd.read_csv(os.path.join(mmlu_data_dir, "test", subject + "_test.csv"), header=None)
            dev_df = dev_df[:min(k, len(dev_df))]
            k = min(k, len(dev_df))

        for i in range(k):
            prompt = gen_prompt(dev_df, subject, i)
            answer = " " + dev_df.iloc[i, dev_df.shape[1] - 2 + 1]

            if use_chat_format:
                if chat_format == "tulu":
                    prompt = "<|user|>\n" + prompt + "\n<|assistant|>\nThe answer is:"
                else:
                    # f"<s> {B_INST} {task_prompt.strip()} {question} {E_INST} A:"
                    prompt = f"<s> {B_INST} {prompt} {E_INST} The answer is:"
            else:
                prompt = prompt
            full_input_ids, labels, attention_mask = tokenize(tokenizer, prompt, answer, max_length, max_res_length, print_ex=True if i == 0 else False)
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)
    dataset = Dataset.from_dict(dataset)
    return dataset


def get_dataset(task, **kwargs):
    """
    Get the dataset for the given task.

    Args:
        task_name (str): The name of the task.

    Raises:
        ValueError: If the task name is not valid.

    Returns:
        Dataset: The dataset.
    """
    if task == "bbh":
        return get_bbh_dataset(**kwargs)
    elif task == "tydiqa":
        return get_tydiqa_dataset(**kwargs)
    elif task == "mmlu":
        return get_mmlu_dataset(**kwargs)
    else:
        raise ValueError("Invalid task name")


def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,  # When getting gradients, we only do this single batch process
                            collate_fn=data_collator)
    print("There are {} examples in the dataset".format(len(dataset)))
    return dataloader
