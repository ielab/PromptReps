import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from tevatron.retriever.dataset import EncodeDataset, TrainDataset
from datasets import load_dataset
from torch.utils.data import Dataset
from typing import Tuple, List
from dataclasses import dataclass
from arguments import PromptRepsDataArguments
import random

logger = logging.getLogger(__name__)


class PromptRepsEncodeDataset(EncodeDataset):
    def __getitem__(self, item) -> Tuple[str, str]:
        text = self.encode_data[item]
        if self.data_args.encode_is_query:
            text_id = text['query_id']
            formated_text = text['query'].strip()
        else:
            text_id = text['docid']
            formated_text = f"{text['title']} {text['text']}".strip()
        return text_id, formated_text


@dataclass
class PromptRepsEncodeCollator:
    data_args: PromptRepsDataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, str]]):
        """
        Collate function for encoding.
        :param features: list of (id, text) tuples
        """
        text_ids = [x[0] for x in features]
        texts = [x[1] for x in features]
        max_length = self.data_args.query_max_len if self.data_args.encode_is_query else self.data_args.passage_max_len
        collated_texts = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        prefix = self.data_args.query_prefix if self.data_args.encode_is_query else self.data_args.passage_prefix
        suffix = self.data_args.query_suffix if self.data_args.encode_is_query else self.data_args.passage_suffix

        collated_texts['input_ids'] = [self.tokenizer.encode(prefix, add_special_tokens=False) + input_ids +
                                       self.tokenizer.encode(suffix, add_special_tokens=False)
                                       for input_ids in collated_texts['input_ids']]

        collated_texts = self.tokenizer.pad(
            collated_texts,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return text_ids, collated_texts, texts


class PromptRepsTrainDataset(Dataset):
    def __init__(self, data_args: PromptRepsDataArguments, dataset: Dataset = None, trainer=None):
        self.data_args = data_args
        if dataset is not None:
            self.train_data = dataset
        else:
            self.train_data = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config,
                data_files=self.data_args.dataset_path,
                split=self.data_args.dataset_split,
                cache_dir=self.data_args.dataset_cache_dir,
            )

        self.trainer = trainer

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        query = group['query']
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        formated_query = query
        formated_passages = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]

        formated_passages.append(f"{pos_psg['title']} {pos_psg['text']}")

        negative_size = self.data_args.train_group_size - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_group_size == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            formated_passages.append(f"{neg_psg['title']} {neg_psg['text']}")

        return formated_query, formated_passages


@dataclass
class PromptRepsTrainCollator:
    data_args: PromptRepsDataArguments
    tokenizer: PreTrainedTokenizer

    def __call__(self, features: List[Tuple[str, List[str]]]):
        """
        Collate function for training.
        :param features: list of (query, passages) tuples
        :return: tokenized query_ids, passage_ids
        """
        all_queries = [f[0] for f in features]
        all_passages = []
        for f in features:
            all_passages.extend(f[1])
        q_collated = self.tokenizer(
            all_queries,
            padding=False,
            truncation=True,
            max_length=self.data_args.query_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        d_collated = self.tokenizer(
            all_passages,
            padding=False,
            truncation=True,
            max_length=self.data_args.passage_max_len,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )

        q_collated['input_ids'] = [self.tokenizer.encode(self.data_args.query_prefix, add_special_tokens=False)
                                   + input_ids
                                   + self.tokenizer.encode(self.data_args.query_suffix, add_special_tokens=False)
                                   for input_ids in q_collated['input_ids']]
        d_collated['input_ids'] = [self.tokenizer.encode(self.data_args.passage_prefix, add_special_tokens=False)
                                   + input_ids
                                   + self.tokenizer.encode(self.data_args.passage_suffix, add_special_tokens=False)
                                   for input_ids in d_collated['input_ids']]

        q_collated = self.tokenizer.pad(
            q_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        d_collated = self.tokenizer.pad(
            d_collated,
            padding=True,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return q_collated, d_collated