import json
from transformers import AutoTokenizer
import torch
from datasets import load_dataset


class DomainAttribute:

    def __init__(self):
        with open('./data/classes.json', 'r', encoding='utf-8') as f:
            self.classes = json.loads(f.read())
            self.domains = self.classes['domains']
            self.attributes = {tuple(v): int(k) for k, v in self.classes['attributes'].items()}

    def domain2id(self, domain):
        return self.domains[domain]

    def attribute2id(self, attribute):
        return self.attributes[attribute]

    def id2attribute(self,id):
        return self.classes['attributes'][str(id)]

    def num_labels(self):
        return len(self.attributes)


def preprocess_function(example, idx, split, domain_attribute, tokenizer):

    entity = tokenizer.tokenize(example['value'])
    oritext = example["originalContent"]

    tokens = [" <ent> "] + entity + [" <ent> "] + tokenizer.tokenize(oritext)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    domain_id = domain_attribute.domain2id(example['domain'])
    attribute_id = domain_attribute.attribute2id((example['domain'], example['attribute']))

    result = {
        "doc_id": example["id"],
        "input_ids": input_ids,
        "entity_span": len(entity)+1,
        "domain_id": domain_id,
        "attribute_id": attribute_id,
        "entity": entity,
        "split":split,

    }
    return result


def load_datasets(data_args, tokenizer):
    full_datasets = load_dataset("csv", data_files={'train': data_args.train_file,
                                                    'validation': data_args.validation_file,
                                                    'test': data_args.test_file}, cache_dir="../cache/")

    domain_attribute = DomainAttribute()

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
    domain_ids = []
    attribute_ids = []
    entity_spans = []

    for example in examples:
        input_ids.append(torch.LongTensor(example['input_ids']))
        domain_ids.append(example['domain_id'])
        attribute_ids.append(example['attribute_id'])
        entity_spans.append(example['entity_span'])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=VOCAB_PAD)
    domain_ids = torch.LongTensor(domain_ids)
    attribute_ids = torch.LongTensor(attribute_ids)
    entity_spans = torch.LongTensor(entity_spans)

    result = {
        'input_ids': input_ids,
        'attention_mask': input_ids != VOCAB_PAD,
        'labels': attribute_ids,
        'entity_span': entity_spans,

    }
    return result
