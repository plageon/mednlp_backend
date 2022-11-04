import json
import os
import random
import sys

from entity_typing.argparser import ModelArguments, DataTrainingArguments, NERArguments
from entity_typing.collect_data import DomainAttribute
from entity_typing.model import MyRobertaModel

sys.path.append('.')
import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AlbertForSequenceClassification, \
    BertForTokenClassification, AlbertForTokenClassification, HfArgumentParser, TrainingArguments, AutoTokenizer, \
    AutoConfig

from cblue.data import STSDataProcessor, STSDataset, QICDataset, QICDataProcessor, QQRDataset, \
    QQRDataProcessor, QTRDataset, QTRDataProcessor, CTCDataset, CTCDataProcessor, EEDataset, EEDataProcessor
from cblue.trainer import STSTrainer, QICTrainer, QQRTrainer, QTRTrainer, CTCTrainer, EETrainer
from cblue.utils import init_logger, seed_everything
import numpy as np

__all__ = ["do_ner"]

_g = None


class NER_MODEL():
    def __init__(self):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file("train_args.json")
        ner_args_parser = HfArgumentParser(NERArguments)
        ner_args = ner_args_parser.parse_json_file("ner_args.json")[0]

        # load ner trainer
        dataset_class, data_processor_class = EEDataset, EEDataProcessor
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification

        trainer_class = EETrainer
        ngram_dict = None

        logger = init_logger(os.path.join(ner_args.output_dir, f'ee_{ner_args.model_name}.log'))
        device = torch.device("cpu")
        ner_args.device = device

        self.mac_tokenizer = tokenizer_class.from_pretrained(ner_args.output_dir)
        data_processor = data_processor_class(root=ner_args.data_dir)
        self.data_processor = data_processor
        self.ner_args = ner_args
        self.no_entity_label = '0'

        # test_samples = data_processor.get_test_sample()

        self.ner_model = model_class.from_pretrained(ner_args.output_dir, num_labels=data_processor.num_labels)
        self.ner_trainer = trainer_class(args=ner_args, model=self.ner_model, data_processor=self.data_processor,
                                         tokenizer=self.mac_tokenizer, logger=logger, model_class=model_class,
                                         ngram_dict=ngram_dict)

        # load entity typing model

        # model_args.model_name_or_path='attribute'
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        self.roberta_tokenizer.add_tokens("<ent>")
        self.domain_attribute = DomainAttribute()
        num_labels = self.domain_attribute.num_labels()
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        self.entity_typing_model = MyRobertaModel(config, num_labels, model_name_or_path=model_args.model_name_or_path)
        self.entity_typing_model.load_state_dict(torch.load('attribute/pytorch_model.bin', map_location=device))
        self.ner_model.eval()

    def preprocess_function(self, examples):
        VOCAB_PAD = 0
        LABEL_PAD = -100

        input_ids = []
        entity_spans = []

        for example in examples:
            oritext = example["text"]
            for entity_item in example['entities']:
                entity = self.roberta_tokenizer.tokenize(entity_item['entity'])

                tokens = [" <ent> "] + entity + [" <ent> "] + self.roberta_tokenizer.tokenize(oritext)
                input_ids.append(torch.LongTensor(self.roberta_tokenizer.convert_tokens_to_ids(tokens)))
                entity_spans.append(len(entity) + 1)
        if not input_ids:
            return None

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=VOCAB_PAD)
        entity_spans = torch.LongTensor(entity_spans)

        result = {
            'input_ids': input_ids,
            'attention_mask': input_ids != VOCAB_PAD,
            'entity_span': entity_spans,

        }
        return result

    def do_entity_typing(self, texts):
        inputs = self.preprocess_function(texts)
        if not inputs:
            return None
        self.entity_typing_model.eval()
        outputs = self.entity_typing_model(**inputs)['logits'].detach().numpy()
        pred_attribute_ids = np.argmax(outputs, axis=1)
        pred_attributes = [self.domain_attribute.id2attribute(i) for i in pred_attribute_ids]
        cnt = 0
        for text in texts:
            for entity in text['entities']:
                entity['type'] = pred_attributes[cnt]
                cnt += 1

        # print(texts)
        return texts

    def preprocess(self, samples, is_predict=True):
        def label_data(data, start, end, _type):
            """label_data"""
            for i in range(start, end + 1):
                suffix = "B-" if i == start else "I-"
                data[i] = "{}{}".format(suffix, _type)
            return data

        outputs = {'text': [], 'label': [], 'orig_text': []}
        for data in samples:
            text_a = ["，" if t == " " or t == "\n" or t == "\t" else t
                      for t in list(data["text"].lower())]
            # text_a = "\002".join(text_a)
            outputs['text'].append(text_a)
            outputs['orig_text'].append(data['text'])
            if not is_predict:
                labels = [self.no_entity_label] * len(text_a)
                for entity in data['entities']:
                    start_idx, end_idx, type = entity['start_idx'], entity['end_idx'], entity['type']
                    labels = label_data(labels, start_idx, end_idx, type)
                outputs['label'].append('\002'.join(labels))
        return outputs

    def do_ner(self, texts):
        dataset_class, data_processor_class = EEDataset, EEDataProcessor
        snts = []
        num_snts = [0]
        for text in texts:
            num_snts.append(num_snts[-1])
            for snt in text['text'].split('。'):
                if snt:
                    snts.append({
                        'text': snt
                    })
                    num_snts[-1] += 1

        test_samples = self.preprocess(snts)
        test_dataset = dataset_class(test_samples, self.data_processor, self.mac_tokenizer, mode='test',
                                     ngram_dict=None,
                                     max_length=self.ner_args.max_length, model_type=self.ner_args.model_type)

        ner_res = self.ner_trainer.predict(test_dataset=test_dataset, model=self.ner_model, save_result=False)
        for snt, snt_entities in zip(snts, ner_res):
            for snt_entity in snt_entities:
                snt_entity['sentence'] = snt['text']+'。'

        for i, text in enumerate(texts):
            text['entities'] = ner_res[num_snts[i]:num_snts[i + 1]]
        # res = self.do_entity_typing(texts)
        return texts

    def do_train(self, docs):
        ner_args_parser = HfArgumentParser(NERArguments)
        ner_args = ner_args_parser.parse_json_file("ner_args.json")[0]

        # load ner trainer
        dataset_class, data_processor_class = EEDataset, EEDataProcessor
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification

        trainer_class = EETrainer
        ngram_dict = None

        logger = init_logger(os.path.join(ner_args.output_dir, f'ee_{ner_args.model_name}.log'))
        device = torch.device("cpu")
        ner_args.device = device

        self.mac_tokenizer = tokenizer_class.from_pretrained(ner_args.output_dir)
        data_processor = data_processor_class(root=ner_args.data_dir)
        self.data_processor = data_processor
        self.ner_args = ner_args

        random.shuffle(docs)
        split_point = len(docs) // 10
        train_texts = docs[:split_point * 8]
        eval_texts = docs[split_point * 8:split_point * 9]
        test_texts = docs[split_point * 9:]
        train_samples = self.preprocess(train_texts, is_predict=False)
        eval_samples = self.preprocess(eval_texts, is_predict=False)
        test_samples = self.preprocess(test_texts, is_predict=False)
        train_dataset = dataset_class(train_samples, self.data_processor, self.mac_tokenizer, mode='train',
                                      ngram_dict=None,
                                      max_length=self.ner_args.max_length, model_type=self.ner_args.model_type)
        eval_dataset = dataset_class(eval_samples, self.data_processor, self.mac_tokenizer, mode='eval',
                                     ngram_dict=None,
                                     max_length=self.ner_args.max_length, model_type=self.ner_args.model_type)
        test_dataset = dataset_class(test_samples, self.data_processor, self.mac_tokenizer, mode='eval',
                                     ngram_dict=None,
                                     max_length=self.ner_args.max_length, model_type=self.ner_args.model_type)

        self.ner_trainer = trainer_class(args=ner_args, model=self.ner_model, data_processor=self.data_processor,
                                         tokenizer=self.mac_tokenizer, logger=logger, train_dataset=train_dataset,
                                         eval_dataset=eval_dataset, model_class=model_class,
                                         ngram_dict=ngram_dict)

        global_step, best_step = self.ner_trainer.train()
        self.ner_trainer.evaluate(self.ner_trainer.model, do_test=True, test_dataset=test_dataset)
        print(global_step, best_step)
    def do_predict(self,predict_docs):
        ner_args_parser = HfArgumentParser(NERArguments)
        ner_args = ner_args_parser.parse_json_file("ner_args.json")[0]

        # load ner trainer
        dataset_class, data_processor_class = EEDataset, EEDataProcessor
        tokenizer_class, model_class = BertTokenizer, BertForTokenClassification

        trainer_class = EETrainer
        ngram_dict = None

        logger = init_logger(os.path.join(ner_args.output_dir, f'ee_{ner_args.model_name}.log'))
        device = torch.device("cpu")
        ner_args.device = device

        self.mac_tokenizer = tokenizer_class.from_pretrained(ner_args.output_dir)
        data_processor = data_processor_class(root=ner_args.data_dir)
        self.data_processor = data_processor
        self.ner_args = ner_args
        predict_samples = self.preprocess(predict_docs, is_predict=True)
        test_dataset = dataset_class(predict_samples, self.data_processor, self.mac_tokenizer, mode='test',
                                     ngram_dict=None,
                                     max_length=self.ner_args.max_length, model_type=self.ner_args.model_type)

        self.ner_trainer = trainer_class(args=ner_args, model=self.ner_model, data_processor=self.data_processor,
                                         tokenizer=self.mac_tokenizer, logger=logger, model_class=model_class,
                                         ngram_dict=ngram_dict)
        self.ner_trainer.predict(self.ner_model,test_dataset)
if _g is None:
    _g = NER_MODEL()

def do_ner(texts):
    return _g.do_ner(texts)


if __name__ == '__main__':
    texts = [{'text': '1.(膀胱) 小块粘膜组织呈慢性炎性反应。'}]
    TRAIN = False
    EVALUATE = True
    with open('data/CMeEE/cmee.json', 'r', encoding='utf-8') as f:
        train_eval_docs = json.loads(f.read())

    if TRAIN:
        ner_model = NER_MODEL()

        ner_model.do_train(train_eval_docs)

    if EVALUATE:
        ner_model = NER_MODEL()

        # ner_model.do_predict(texts)
        print(ner_model.do_ner(texts))


