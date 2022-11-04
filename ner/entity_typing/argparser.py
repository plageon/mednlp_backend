from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    span_len: int = field(
        default=8
    )
    max_len: int = field(
        default=512
    )
    meta_file: str = field(
        default=''
    )
    task_name: str = field(
        default=''
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    pos_loss_weight: float = field(
        default=1.0
    )
    span_len_embedding_range: int = field(
        default=50
    )
    span_len_embedding_hidden_size: int = field(
        default=20
    )
    not_bert_learning_rate: float = field(
        default=0.0001
    )
    gcn_layers: int = field(
        default=3
    )
    train_dglgraph_path: str = field(
        default=''
    )
    dev_dglgraph_path: str = field(
        default=''
    )
    test_dglgraph_path: str = field(
        default=''
    )
    train_snt_dglgraph_path: str = field(
        default=''
    )
    dev_snt_dglgraph_path: str = field(
        default=''
    )
    test_snt_dglgraph_path: str = field(
        default=''
    )
    lambda_boundary: float = field(
        default=0.0
    )
    event_embedding_size: int = field(
        default=200
    )

@dataclass
class NERArguments:
    output_dir: str=field(
        default=''
    )
    data_dir: str=field(
        default=''
    )
    model_name: str=field(
        default=''
    )
    model_type:str=field(
        default='bert'
    )
    task_name:str=field(
        default='ee'
    )
    max_length: int = field(default=128, metadata={
        "help": "the max length of sentence."})
    train_batch_size: int = field(default=8, metadata={
        "help": "Batch size for training."})
    eval_batch_size: int = field(default=8, metadata={
        "help": "Batch size for evaluation."})
    learning_rate: float = field(default=5e-5, metadata={
        "help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.01, metadata={
        "help": "Weight deay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={
        "help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={
        "help": "Max gradient norm."})
    epochs: int = field(default=3, metadata={
        "help": "Total number of training epochs to perform."})
    warmup_proportion: float = field(default=0.1, metadata={
        "help": "Proportion of training to perform linear learning rate warmup for, "
                "E.g., 0.1 = 10% of training."})
    earlystop_patience: int = field(default=2, metadata={
        "help": "The patience of early stop"})

    logging_steps: int = field(default=10, metadata={
        "help": "Log every X updates steps."})
    save_steps: int = field(default=1000, metadata={
        "help": "Save checkpoint every X updates steps."})
    seed: int = field(default=2021, metadata={
        "help": "random seed for initialization"})

