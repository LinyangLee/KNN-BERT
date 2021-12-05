# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script_v0 on your own text classification task. Pointers for this are left as comments.
import dataclasses
import inspect
import logging
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import sys
import torch
import fitlog
from dataclasses import dataclass, field
from sklearn.metrics import matthews_corrcoef
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric
from glue import Glue

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    EvalPrediction,
    set_seed,
)
from my_trainer import Trainer, MyEvalPrediction
from training_args import TrainingArguments
from model import (
    ContrastiveMoCo,
    ContrastiveRoBertaMoCo,
    ContrastiveMoCoKnnBert,
    ContrastiveRobertaMoCoKnnBert,
    SCLModel,
    SCLRoBertaModel
)

from transformers.trainer_utils import is_main_process

from transformers import AutoModelForSequenceClassification

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    test_file: Optional[str] = field(
        default=None,
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


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

    train_pattern: str = field(default="further_pretrain")

@dataclass
class FitLogArguments:
    task: str
    negative_num: int = field(default=96)
    positive_num: int = field(default=16)
    queue_size: int = field(default=32000)
    top_k: int = field(default=100)
    end_k: int = field(default=1)
    contrastive_rate_in_training: float = field(default=0.1)
    contrastive_rate_in_inference: float = field(default=0.1)

    memory_bank: bool = field(default=False)
    random_positive: bool = field(default=False)


def data_collator(features):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    first = features[0]
    batch = {}
    # if "original_text" in first:
    #     batch["original_text"] = [f["original_text"] for f in features]
    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script_v0.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, FitLogArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script_v0 and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, fitlog_args = parser.parse_args_into_dataclasses()

    training_args.sample_file = data_args.train_file
    training_args.max_length = data_args.max_seq_length
    fitlog.set_log_dir(training_args.fitlog_dir)
    fitlog_args_dict = {"seed": training_args.seed,
                        "warmup_steps": training_args.warmup_steps,
                        "lr": training_args.learning_rate}
    fitlog_args_name = [i for i in dir(fitlog_args) if i[0] != "_"]
    for args_name in fitlog_args_name:
        args_value = getattr(fitlog_args, args_name)
        training_args.__dict__[args_name] = args_value
        if args_value is not None:
            fitlog_args_dict[args_name] = args_value
    fitlog.add_hyper(fitlog_args_dict)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script_v0 will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script_v0 does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        if training_args.do_predict:
            datasets = load_dataset(
                "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file,
                                   "test": data_args.test_file}, sep="\t"
            )
        else:
            datasets = load_dataset(
                "csv", data_files={"train": data_args.train_file, "validation": data_args.validation_file}, sep="\t"
            )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json", data_files={"train": data_args.train_file, "validation": data_args.validation_file}
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if training_args.load_trained_model:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

        config.model_name = model_args.model_name_or_path
        config.negative_num = training_args.negative_num
        config.positive_num = training_args.positive_num
        config.queue_size = training_args.queue_size
        config.train_multi_head = training_args.train_multi_head
        config.contrastive_rate_in_training = training_args.contrastive_rate_in_training
        config.load_trained_model = training_args.load_trained_model
        config.multi_head_num = training_args.multi_head_num
        config.memory_bank = training_args.memory_bank
        config.random_positive = training_args.random_positive

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path
        )
        if training_args.load_model_pattern == "original_model":
            model = AutoModelForSequenceClassification.from_pretrained(
                training_args.model_path,
                config=config
            )
        elif training_args.load_model_pattern == "moco":
            model = ContrastiveMoCo(config=config)
        elif training_args.load_model_pattern == "roberta_moco":
            model = ContrastiveRoBertaMoCo(config=config)
        elif training_args.load_model_pattern == "knn_bert":
            config.knn_num = training_args.top_k
            model = ContrastiveMoCoKnnBert(config=config)
            logger.info("loading model form " + training_args.model_path + "pytorch_model.bin")
            state_dict = torch.load(training_args.model_path + "pytorch_model.bin")
            model.load_state_dict(state_dict)
        elif training_args.load_model_pattern == "knn_roberta":
            config.knn_num = training_args.top_k
            config.end_k = training_args.end_k
            model = ContrastiveRobertaMoCoKnnBert(config=config)
            logger.info("loading model form " + training_args.model_path + "pytorch_model.bin")
            state_dict = torch.load(training_args.model_path + "pytorch_model.bin")
            model.load_state_dict(state_dict)
        else:
            logger.warning("your model should in list [original_model moco roberta_moco knn_bert knn_roberta]")
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
        )

        config.model_name = model_args.model_name_or_path
        config.negative_num = training_args.negative_num
        config.positive_num = training_args.positive_num
        config.queue_size = training_args.queue_size
        config.train_multi_head = training_args.train_multi_head
        config.contrastive_rate_in_training = training_args.contrastive_rate_in_training
        config.load_trained_model = training_args.load_trained_model
        config.multi_head_num = training_args.multi_head_num
        config.memory_bank = training_args.memory_bank
        config.random_positive = training_args.random_positive

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        if training_args.load_model_pattern == "original_model":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config
            )
        elif training_args.load_model_pattern == "moco":
            model = ContrastiveMoCo(config=config)
        elif training_args.load_model_pattern == "knn_bert":
            config.knn_num = training_args.top_k
            config.end_k = training_args.end_k
            model = ContrastiveMoCoKnnBert(config=config)
        elif training_args.load_model_pattern == "knn_roberta":
            config.knn_num = training_args.top_k
            config.end_k = training_args.end_k
            model = ContrastiveRobertaMoCoKnnBert(config=config)
        elif training_args.load_model_pattern == "roberta_moco":
            model = ContrastiveRoBertaMoCo(config=config)
        elif training_args.load_model_pattern == "scl_model":
            model = SCLModel.from_pretrained(
                model_args.model_name_or_path,
                config=config
            )
        elif training_args.load_model_pattern == "scl_model_roberta":
            model = SCLRoBertaModel.from_pretrained(
                model_args.model_name_or_path,
                config=config
            )
        # elif training_args.load_model_pattern == "knn_init_by_bert":
        #     config.knn_num = training_args.top_k
        #     config.end_k = training_args.end_k
        #     model = ContrastiveMoCoKnnInitByBert(
        #         config=config
        #     )
        else:
            logger.warning("your model should in list [original_model moco roberta_moco knn_bert knn_roberta scl_model]")

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
            model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
            and data_args.task_name is not None
            and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        # result["sent_id"] = [index for index, i in enumerate(examples["label"])]
        # result["original_text"] = examples[sentence1_key]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    test_dataset = datasets["test"] if training_args.do_predict else None

    if data_args.task_name is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        # metric = load_metric("glue", data_args.task_name)
        metric = Glue(data_args.task_name)

    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: MyEvalPrediction):

        preds_knn = p.prediction_by_knn[0] if isinstance(p.prediction_by_knn, tuple) else p.prediction_by_knn
        preds_cls = p.prediction_by_cls[0] if isinstance(p.prediction_by_cls, tuple) else p.prediction_by_cls
        preds_com = p.prediction_combine[0] if isinstance(p.prediction_combine, tuple) else p.prediction_combine
        preds_knn = np.squeeze(preds_knn) if is_regression else np.argmax(preds_knn, axis=1)
        preds_cls = np.squeeze(preds_cls) if is_regression else np.argmax(preds_cls, axis=1)
        preds_com = np.squeeze(preds_com) if is_regression else np.argmax(preds_com, axis=1)

        if data_args.task_name is not None:

            result_knn = metric.compute(predictions=preds_knn, references=p.label_ids)
            result_cls = metric.compute(predictions=preds_cls, references=p.label_ids)
            result_com = metric.compute(predictions=preds_com, references=p.label_ids)
            result = {
                "knn_" + list(result_knn.keys())[-1]: list(result_knn.values())[-1],
                "cls_" + list(result_cls.keys())[-1]: list(result_cls.values())[-1],
                "com_" + list(result_com.keys())[-1]: list(result_com.values())[-1]
            }
            return result

        return {
            "accuracy_knn": (preds_knn == p.label_ids).astype(np.float32).mean().item(),
            "accuracy_cls": (preds_cls == p.label_ids).astype(np.float32).mean().item(),
            "accuracy_com": (preds_com == p.label_ids).astype(np.float32).mean().item(),
        }

    # Initialize our Trainer
    training_args.num_labels = config.num_labels
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        test_dataset=test_dataset if training_args.do_predict else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload
        fitlog.finish()

    return None


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
