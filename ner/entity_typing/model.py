import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, AutoModel


class MyRobertaModel(PreTrainedModel):
    def __init__(self, config, num_labels,model_name_or_path):
        super().__init__(config)
        self.roberta = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, num_labels),
        )
        self.loss_func = CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            domain_id=None,
            attribute_id=None,
            entity_span=None,
            labels=None,
    ):
        batch_size = input_ids.size(0)
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        span_output = []
        for hidden, index in zip(last_hidden_state, entity_span):
            span_output.append(hidden[index] - hidden[0])
        span_output = torch.stack(span_output).view(batch_size, -1)
        logits = self.linear(span_output)
        loss = None
        if labels:
            loss = self.loss_func(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
        }
