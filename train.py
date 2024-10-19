import logging
import os
import sys

from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
    is_torch_xla_available,
    EarlyStoppingCallback,
    EvalPrediction
)
from tevatron.retriever.arguments import ModelArguments
from arguments import PromptRepsDataArguments as DataArguments, \
    PromptRepsTrainingArguments as TrainingArguments
from dataset import PromptRepsTrainDataset as TrainDataset, \
    PromptRepsTrainCollator as TrainCollator
from modeling import PromptRepsLLM, EncoderModel
from tevatron.retriever.gc_trainer import GradCacheTrainer as GCTrainer
from trainer import EarlyCheckpointCallback, PromptRepsTrainer
import torch

logger = logging.getLogger(__name__)


# def make_compute_metrics(model, training_args):  # for eval dense and sparse loss
#     def compute_metrics(eval_preds):
#         predictions = eval_preds.predictions
#         labels = eval_preds.label_ids  # not used for now
#         q_reps, p_reps, q_logits, p_logits = predictions
#         scores_dense = model.compute_similarity(torch.tensor(q_reps, requires_grad=False),
#                                                 torch.tensor(p_reps, requires_grad=False))
#         scores_dense = scores_dense.view(q_reps.shape[0], -1)
#         target = torch.arange(scores_dense.shape[0], device=scores_dense.device, dtype=torch.long)
#         target = target * (p_reps.shape[0] // q_reps.shape[0])
#         loss_dense = model.compute_loss(scores_dense / model.temperature, target).item()
#
#         if training_args.hybrid_training:
#             # sparse loss
#             scores_sparse = model.compute_similarity(torch.tensor(q_logits, requires_grad=False),
#                                                      torch.tensor(p_logits, requires_grad=False))
#             scores_sparse = scores_sparse.view(q_logits.shape[0], -1)
#             loss_sparse = model.compute_loss(scores_sparse, target).item()
#         else:
#             loss_sparse = 0.0
#         return {
#             'loss_dense': loss_dense,
#             'loss_sparse': loss_sparse,
#         }
#     return compute_metrics


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id else tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    model = PromptRepsLLM.build(
        model_args,
        training_args,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = TrainDataset(data_args)
    if training_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), training_args.max_train_samples)
        train_dataset = train_dataset.train_data.select(range(max_train_samples))
        train_dataset = TrainDataset(data_args, dataset=train_dataset)

    eval_dataset = None
    if training_args.do_eval:
        datasets = train_dataset.train_data.train_test_split(training_args.eval_data_percentage)
        train_dataset = datasets['train']
        eval_dataset = datasets['test']
        if training_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), training_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        train_dataset = TrainDataset(data_args, dataset=train_dataset)
        eval_dataset = TrainDataset(data_args, dataset=eval_dataset)

    collator = TrainCollator(data_args, tokenizer)

    # TODO: eval for GCTrainer is not implemented
    if training_args.grad_cache:
        eval_dataset = None
        training_args.do_eval = False

    callbacks = []
    if training_args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience))
    if training_args.save_early_checkpoints:
        callbacks.append(EarlyCheckpointCallback())

    trainer_cls = GCTrainer if training_args.grad_cache else PromptRepsTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        # compute_metrics=make_compute_metrics(model, training_args)
        # if training_args.do_eval and not is_torch_xla_available() else None,
    )
    train_dataset.trainer = trainer
    if eval_dataset:
        eval_dataset.trainer = trainer

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
