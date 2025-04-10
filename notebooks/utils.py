# Import necessary packages
import pandas as pd
import numpy as np
import json
import ast
import subprocess
from pprint import pprint
# import spacy
# import nltk
# from nltk.corpus import stopwords

import re
import logging
import os
import random
import shutil
import sys
import time
# from wordcloud import WordCloud
import networkx as nx

from io import StringIO
import string
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from datetime import datetime
from seqeval.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm, trange

# from spacy.tokens import DocBin
from tqdm import tqdm
import spacy
from spacy.util import filter_spans

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForTokenClassification,
    BertTokenizer,
    CamembertConfig,
    CamembertForTokenClassification,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    XLMRobertaConfig,
    XLMRobertaForTokenClassification,
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.utils.data.distributed import DistributedSampler


from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)



import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



logger = logging.getLogger(__name__)
# Defining a parser which is required for the pipeline
def get_bert_parser():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=False,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=False,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--labels",
        default="",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    return parser


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids

        
def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    # print(1)

    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)
    # print(2)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()


    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "xlnet"] else None
                )  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}
    print(3)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print("results : ",results)
    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="%s-%d".format(mode, guid_index), words=words, labels=labels))
            
    return examples

def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # print('word_tokens : ',word_tokens)
            # print('label : ',label)
            # print('label_map : ',label_map)

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features

def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    logger.info("Creating features from dataset file at %s", args.data_dir)
    examples = read_examples_from_file(args.data_dir, mode)
    features = convert_examples_to_features(
        examples,
        labels,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=bool(args.model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(args.model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(args.model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        pad_token_label_id=pad_token_label_id,
    )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

class BERTmodelTrain:
    """
    Class for training educational attributes
    """
    parser = get_bert_parser()

    original_args = sys.argv
    sys.argv = []
    args = parser.parse_args()
    sys.argv = original_args

    args.model_type = 'bert'
    args.model_name_or_path = 'bert-base-cased'
    args.max_seq_length = 256
    args.num_train_epochs = 6
    args.per_gpu_train_batch_size = 4
    args.save_steps = 750
    args.seed = 1
    # args.labels = 'data/labels.txt'
    args.do_train = True

    # Setup CUDA, GPU & distributed training
    args.device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    args.n_gpu = torch.cuda.device_count()
    # config_class, model_class, tokenizer_class = DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer
    config_class, model_class, tokenizer_class = BertConfig, BertForTokenClassification, BertTokenizer

    # # Prepare CONLL-2003 task
    # labels =['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-MISC', 'I-LOC']
    # num_labels = len(labels)

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    def __init__(self, output_dir, labels, load_model_locally=None, epochs=None):
        if epochs:
            self.args.num_train_epochs = epochs
        self.args.output_dir = output_dir
        self.labels = labels
        self.num_labels = len(self.labels)
        if load_model_locally:
            self.load_saved_model()
        else:
            # loading the pretrained model
            self._load_pretrained_model()
            
    def load_saved_model(self):
        """
        To load the model
        :return:
        """

        labels_ = self.labels
        pad_token_label_id_ = CrossEntropyLoss().ignore_index
        # model_class, tokenizer_class = (DistilBertForTokenClassification, DistilBertTokenizer)
        model_class, tokenizer_class = (BertForTokenClassification, BertTokenizer)

        # if args.do_predict and args.local_rank in [-1, 0]:
        print('\n---Loading model...')
        tm_s = time.time()

        self.tokenizer = tokenizer_class.from_pretrained(self.args.output_dir, do_lower_case=self.args.do_lower_case)
        self.model= model_class.from_pretrained(self.args.output_dir)
        self.model.to(self.args.device)

        tm_e = time.time()
        print('---Model loaded. Time taken is %f seconds...', np.round(tm_e - tm_s, 2))


    def _load_pretrained_model(self):
        print('Loading pretrained model...')
        ts = time.time()


        self.config = BertConfig.from_pretrained(
            self.args.config_name if self.args.config_name else self.args.model_name_or_path,
            num_labels=self.num_labels,
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.tokenizer_name if self.args.tokenizer_name else self.args.model_name_or_path,
            do_lower_case=self.args.do_lower_case,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )
        self.model = BertForTokenClassification.from_pretrained(
            self.args.model_name_or_path,
            from_tf=bool(".ckpt" in self.args.model_name_or_path),
            config=self.config,
            cache_dir=self.args.cache_dir if self.args.cache_dir else None,
        )

        self.model.to(self.args.device)

        te = time.time()
        print('\nPretrained model loaded.  Time taken is %f seconds...',np.round(te - ts, 2))

    def _train(self, data_dir, dump_model=True):
        self.args.data_dir = data_dir
        train_dataset = load_and_cache_examples(self.args, self.tokenizer, self.labels, self.pad_token_label_id,
                                                mode="train")
        train(self.args, train_dataset, self.model, self.tokenizer, self.labels, self.pad_token_label_id)

        # save and upload the model to minio
        if dump_model:
            self._save_the_model()

    def _save_the_model(self):

        if os.path.exists(self.args.output_dir) and os.listdir(self.args.output_dir):
            shutil.rmtree(self.args.output_dir)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)

        logger.info("Saving model checkpoint to %s", self.args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.output_dir, "training_args.bin"))
        
    def _predict(self):
        education_output = {}
        print("""os.path.join(self.args.data_dir, 'test.txt') """,os.path.join(self.args.data_dir, 'test.txt'))
        file = open(os.path.join(self.args.data_dir, 'test.txt'), 'r')
        actual_values_text = file.read()
        file.close()

        tokens_and_labels_list = actual_values_text.split('\n')

        try:
            results, predictions = evaluate(self.args, self.model, self.tokenizer, self.labels, self.pad_token_label_id,
                                      mode='test')
            print("results : ",results)
        except Exception as error_message:
            print("In the evaluation part %s", str(error_message))

        try:
            prediction_text = ""
            sequence_id = 0
            for token_and_label in tokens_and_labels_list:
                if token_and_label.startswith("-DOCSTART-") or token_and_label == "" or token_and_label == "\n":
                    prediction_text += '\n'
                    if not predictions[sequence_id]:
                        sequence_id += 1
                elif predictions[sequence_id]:
                    output_line = token_and_label.split()[0] + " " + predictions[sequence_id].pop(0) + "\n"
                    prediction_text += output_line

                else:
                    # print("Maximum sequence length exceeded: No prediction for '%s'.",
                    #                   token_and_label.split()[0])
                    output_line = token_and_label.split()[0] + " O\n"
                    prediction_text += output_line
            prediction_text = prediction_text[:-1]
        except Exception as error_message:
            print("In prediction section", str(error_message))


        return actual_values_text, prediction_text

    def _predict_on_file(self, file_path):
        education_output = {}
        print("""file_path : """, os.path.join(file_path))
        file = open(file_path, 'r')
        # file = open(os.path.join(self.args.data_dir, 'test.txt'), 'r')
        actual_values_text = file.read()
        file.close()

        tokens_and_labels_list = actual_values_text.split('\n')

        mode = file_path.split('/')[-1].split('.txt')[0]
        print("mode : ", mode)
        try:
            results, predictions = evaluate(self.args, self.model, self.tokenizer, self.labels, self.pad_token_label_id,
                                            mode=mode)
            print("results : ", results)
        except Exception as error_message:
            print("In the evaluation part %s", str(error_message))

        try:
            prediction_text = ""
            sequence_id = 0
            for token_and_label in tokens_and_labels_list:
                if token_and_label.startswith("-DOCSTART-") or token_and_label == "" or token_and_label == "\n":
                    prediction_text += '\n'
                    if not predictions[sequence_id]:
                        sequence_id += 1
                elif predictions[sequence_id]:
                    output_line = token_and_label.split()[0] + " " + predictions[sequence_id].pop(0) + "\n"
                    prediction_text += output_line

                else:
                    # print("Maximum sequence length exceeded: No prediction for '%s'.",
                    #                   token_and_label.split()[0])
                    output_line = token_and_label.split()[0] + " O\n"
                    prediction_text += output_line
            prediction_text = prediction_text[:-1]
        except Exception as error_message:
            print("In prediction section", str(error_message))

        return actual_values_text, prediction_text


