import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoModel


class MyRobertaModel(PreTrainedModel):
    def __init__(self, config, num_period0, num_period1, num_cancer_type, model_name_or_path):
        super().__init__(config)
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.period0_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_period0),
        )
        self.period1_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_period1),
        )
        self.cancer_type_linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_cancer_type),
        )
        self.loss_func = CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            # period0_labels=None,
            # period1_labels=None,
            # cancer_type_labels=None,
    ):
        period0_labels, period1_labels, cancer_type_labels = labels
        batch_size = input_ids.size(0)
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        cls_outputs = last_hidden_state[:, 0]
        period0_logits = self.period0_linear(cls_outputs)
        period1_logits = self.period1_linear(cls_outputs)
        cancer_type_logits = self.cancer_type_linear(cls_outputs)
        loss = None
        if period0_labels is not None:
            period0_loss = self.loss_func(period0_logits, period0_labels)
            period1_loss = self.loss_func(period1_logits, period1_labels)
            cancer_type_loss = self.loss_func(cancer_type_logits, cancer_type_labels)
            loss = period0_loss + period1_loss + cancer_type_loss

        return {
            'loss': loss,
            # 'period0_logits': period0_logits,
            # 'period1_logits': period1_logits,
            # 'cancer_type_logits': cancer_type_logits,
            'logits': [period0_logits, period1_logits, cancer_type_logits],
        }
