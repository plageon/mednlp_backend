import json
import torch
from datasets import load_dataset


class LabelClusters:

    def __init__(self):
        with open('./data/period_type_data/cancer_types.dict', 'r', encoding='utf-8') as f:
            self.cancer_types2id = json.loads(f.read())
        self.id2cancer_types = {v: k for k, v in self.cancer_types2id.items()}

        with open('./data/period_type_data/period0s.dict', 'r', encoding='utf-8') as f:
            self.period0s2id = json.loads(f.read())
        self.id2period0s = {v: k for k, v in self.period0s2id.items()}

        with open('./data/period_type_data/period1s.dict', 'r', encoding='utf-8') as f:
            self.period1s2id = json.loads(f.read())
        self.id2period1s = {v: k for k, v in self.period1s2id.items()}


def preprocess_function(example, idx, split, domain_attribute, tokenizer):
    oritext = example["text"]

    tokens = tokenizer.tokenize(oritext)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    period0_id = domain_attribute.period0s2id[example['period0']]
    period1_id = domain_attribute.period1s2id[example['period1']]
    cancer_type_id = domain_attribute.cancer_types2id[example['cancer_type']]

    result = {
        "doc_id": example["id"],
        "input_ids": input_ids,
        "cancer_type_id": cancer_type_id,
        "period0_id": period0_id,
        "period1_id": period1_id,

    }
    return result


def load_datasets(data_args, tokenizer):
    full_datasets = load_dataset("json", data_files={'train': data_args.train_file,
                                                     'validation': data_args.validation_file,
                                                     'test': data_args.test_file}, cache_dir="./cache/")
    full_datasets.cleanup_cache_files()

    domain_attribute = LabelClusters()

    preprocess_args = {'split': 'train', 'domain_attribute': domain_attribute, 'tokenizer': tokenizer}
    train_dataset = full_datasets["train"].map(preprocess_function, batched=False, with_indices=True,
                                               load_from_cache_file=not data_args.overwrite_cache,
                                               fn_kwargs=preprocess_args)
    eval_dataset = full_datasets["validation"].map(preprocess_function, batched=False, with_indices=True,
                                                   load_from_cache_file=not data_args.overwrite_cache,
                                                   fn_kwargs=preprocess_args)
    test_dataset = full_datasets["test"].map(preprocess_function, batched=False, with_indices=True,
                                             load_from_cache_file=not data_args.overwrite_cache,
                                             fn_kwargs=preprocess_args)

    return train_dataset, eval_dataset, test_dataset


def collator_fn(examples):
    VOCAB_PAD = 0
    LABEL_PAD = -100

    input_ids = []
    period0_ids = []
    period1_ids = []
    cancer_type_ids = []

    for example in examples:
        input_ids.append(torch.LongTensor(example['input_ids']))
        period0_ids.append(example['period0_id'])
        period1_ids.append(example['period1_id'])
        cancer_type_ids.append(example['cancer_type_id'])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=VOCAB_PAD)
    period0_ids = torch.LongTensor(period0_ids)
    period1_ids = torch.LongTensor(period1_ids)
    cancer_type_ids = torch.LongTensor(cancer_type_ids)

    result = {
        'input_ids': input_ids,
        'attention_mask': input_ids != VOCAB_PAD,
        'labels': [period0_ids, period1_ids, cancer_type_ids],
        # "cancer_type_labels": cancer_type_ids,
        # "period0_labels": period0_ids,
        # "period1_labels": period1_ids,

    }
    return result
