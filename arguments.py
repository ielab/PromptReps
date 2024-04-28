import logging
from tevatron.retriever.arguments import ModelArguments, DataArguments, TevatronTrainingArguments
from dataclasses import dataclass, field
import os
logger = logging.getLogger(__name__)


@dataclass
class PromptRepsDataArguments(DataArguments):
    query_suffix: str = field(
        default='', metadata={"help": "suffix or instruction for query"}
    )
    passage_suffix: str = field(
        default='', metadata={"help": "suffix or instruction for passage"}
    )
    dense_output_dir: str = field(default=None, metadata={"help": "where to save the encode dense vectors"})
    sparse_output_dir: str = field(default=None, metadata={"help": "where to save the encode dense vectors"})
    sparse_exact_match: bool = field(default=True, metadata={"help": "whether to use exact match for sparse retrieval"})

    def __post_init__(self):
        if os.path.exists(self.query_prefix):
            with open(self.query_prefix, 'r') as f:
                self.query_prefix = f.read().strip()

        if os.path.exists(self.query_suffix):
            with open(self.query_suffix, 'r') as f:
                self.query_suffix = f.read().strip()

        if os.path.exists(self.passage_prefix):
            with open(self.passage_prefix, 'r') as f:
                self.passage_prefix = f.read().strip()

        if os.path.exists(self.passage_suffix):
            with open(self.passage_suffix, 'r') as f:
                self.passage_suffix = f.read().strip()


@dataclass
class PromptRepsTrainingArguments(TevatronTrainingArguments):
    q_flops_loss_factor: float = field(default=0.01)
    p_flops_loss_factor: float = field(default=0.01)
    eval_data_percentage: float = field(default=0.1)
    max_eval_samples: int = field(default=2000)
