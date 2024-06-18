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
    num_expended_tokens: int = field(default=0, metadata={"help": "Number of expended tokens. Default is 0, "
                                                                  "meaning exact term matching only."})
    num_pooled_tokens: int = field(default=0, metadata={"help": "Number of tokens to form the embeddings."})
    multi_reps: bool = field(default=False, metadata={"help": "Whether to use multiple representations for retrieval (ColBERT style)"})
    word_level_reps: bool = field(default=False, metadata={"help": "Whether to use word level representations for retrieval"})

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
    max_eval_samples: int = field(default=None)
    max_train_samples: int = field(default=None)
    save_early_checkpoints: bool = field(default=False)
    hybrid_training: bool = field(default=False)
    early_stopping_patience: int = field(default=None)


@dataclass
class PromptRepsSearchArguments:
    passage_reps: str = field(default=None, metadata={"help": "Path to passage dense representations"})
    sparse_index: str = field(default=None, metadata={"help": "Path to passage sparse representations"})
    depth: int = field(default=1000)
    save_dir: str = field(default=None, metadata={"help": "Where to save the run files"})
    quiet: bool = field(default=True, metadata={"help": "Whether to print the progress"})
    use_gpu: bool = field(default=False, metadata={"help": "Whether to use GPU"})
    alpha: float = field(default=0.5, metadata={"help": "The weight for dense retrieval"})
    batch_size: int = field(default=128, metadata={"help": "Batch size for retrieval"})
    remove_query: bool = field(default=False, metadata={"help": "Whether to remove query id from the ranking"})
    threads: int = field(default=1, metadata={"help": "Number of threads for sparse retrieval"})



