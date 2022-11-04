import os

import torch
from transformers import HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer
import numpy as np

from ner.period_type.argparser import ModelArguments, DataTrainingArguments
from ner.period_type.model import MyRobertaModel
from ner.period_type.collect_data import LabelClusters, load_datasets, collator_fn
from ner.period_type.trainer import MyTrainer
from ner.period_type.util import compute_metrics

__all__ = ["do_ner"]

_g = None


class PERIOD_TYPE_MODEL():
    def __init__(self, ):
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        self.model_args, self.data_args, self.training_args = parser.parse_json_file("period_type_args.json")
        device = torch.device("cpu")
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        self.compute_metrics = compute_metrics
        labelcluster = LabelClusters()
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(self.model_args.model_name_or_path)

        self.model = MyRobertaModel(config, len(labelcluster.period0s2id), len(labelcluster.period1s2id),
                                    len(labelcluster.cancer_types2id),
                                    model_name_or_path=self.model_args.model_name_or_path)
        self.model.load_state_dict(torch.load('attribute/pytorch_model.bin', map_location=device))
        # self.model.from_pretrained(self.training_args.output_dir)

    def train_period_type(self):
        train_dataset, eval_dataset, test_dataset = load_datasets(self.data_args, self.roberta_tokenizer)
        trainer = MyTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset if self.training_args.do_train else None,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            tokenizer=self.roberta_tokenizer,
            data_collator=collator_fn,
            not_bert_learning_rate=self.model_args.not_bert_learning_rate,
        )
        # Training
        last_checkpoint = None
        if self.training_args.do_train:
            checkpoint = None
            if last_checkpoint is not None:
                checkpoint = last_checkpoint
            elif os.path.isdir(self.model_args.model_name_or_path):
                # Check the config from that potential checkpoint has the right number of labels before using it as a
                # checkpoint.
                checkpoint = self.model_args.model_name_or_path

            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            metrics["train_samples"] = len(train_dataset)

            trainer.save_model()  # Saves the tokenizer too for easy upload

            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        if self.training_args.do_eval:
            # on dev
            p = trainer.predict(test_dataset=eval_dataset)
            acc = compute_metrics(p)
            print("accuracy on dev:", acc)

            # on test
            p = trainer.predict(test_dataset=test_dataset)
            acc = compute_metrics(p)
            print("accuracy on test:", acc)

    def preprocess_function(self, examples):
        label_clusters = self.label_clusters



        VOCAB_PAD = 0
        LABEL_PAD = -100

        input_ids = []
        period0_ids = []
        period1_ids = []
        cancer_type_ids = []

        for example in examples:
            oritext = example["text"]

            tokens = self.roberta_tokenizer.tokenize(oritext)
            input_id = self.roberta_tokenizer.convert_tokens_to_ids(tokens)
            # period0_id = label_clusters.period0s2id[example['period0']]
            # period1_id = label_clusters.period1s2id[example['period1']]
            # cancer_type_id = label_clusters.cancer_types2id[example['cancer_type']]
            input_ids.append(torch.LongTensor(input_id))
            # period0_ids.append(period0_id)
            # period1_ids.append(period1_id)
            # cancer_type_ids.append(cancer_type_id)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=VOCAB_PAD)
        # period0_ids = torch.LongTensor(period0_ids)
        # period1_ids = torch.LongTensor(period1_ids)
        # cancer_type_ids = torch.LongTensor(cancer_type_ids)

        result = {
            'input_ids': input_ids,
            'attention_mask': input_ids != VOCAB_PAD,
            'labels': [None,None,None],
            # "cancer_type_labels": cancer_type_ids,
            # "period0_labels": period0_ids,
            # "period1_labels": period1_ids,

        }
        return result

    def do_period_type(self, texts):
        self.label_clusters = LabelClusters()
        inputs = self.preprocess_function(texts)
        if not inputs:
            return None
        self.model.eval()
        outputs = [o.detach().numpy() for o in self.model(**inputs)['logits']]
        pred_attribute_ids = [np.argmax(output, axis=1) for output in outputs]
        pred_period0s = [self.label_clusters.id2period0s[i] for i in pred_attribute_ids[0]]
        pred_period1s = [self.label_clusters.id2period1s[i] for i in pred_attribute_ids[1]]
        pred_cancer_types = [self.label_clusters.id2cancer_types[i] for i in pred_attribute_ids[2]]
        for pred_period0, pred_period1, pred_cancer_type, text in zip(pred_period0s, pred_period1s, pred_cancer_types,
                                                                      texts):
            text['period0'] = pred_period0
            text['period1'] = pred_period1
            text['cancer_type'] = pred_cancer_type

        # print(texts)
        return texts


if _g is None:
    _g = PERIOD_TYPE_MODEL()


def do_period_type(texts):
    return _g.do_period_type(texts)


if __name__ == '__main__':
    texts = [{'text': '1.(膀胱) 小块粘膜组织呈慢性炎性反应。'}]
    period_type_model = PERIOD_TYPE_MODEL()
    # res=period_type_model.do_period_type(texts)
    period_type_model.train_period_type()
    # print(res)
