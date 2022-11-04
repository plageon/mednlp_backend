from transformers import Trainer

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.not_bert_learning_rate = kwargs.pop('not_bert_learning_rate')
        super().__init__(*args, **kwargs)

    def easy_predict(self):
        pass


