import logging
import os
from transformers.trainer import TRAINING_ARGS_NAME, TrainerCallback, TrainerState, TrainerControl, IntervalStrategy
from arguments import PromptRepsDataArguments as DataArguments, \
    PromptRepsTrainingArguments as TrainingArguments
from modeling import PromptRepsLLM, EncoderModel
from tevatron.retriever.trainer import TevatronTrainer as Trainer
from torch import nn
import torch
from typing import Dict, Union, Optional, List, Any, Tuple
from peft import get_peft_model_state_dict

logger = logging.getLogger(__name__)


class EarlyCheckpointCallback(TrainerCallback):
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Save
        if (
                state.global_step == 10
                or state.global_step == 50
                or state.global_step == 100
                or state.global_step == 200
                or state.global_step == 500
        ):
            control.should_save = True
        return control


class PromptRepsTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs,
                        hybrid_training=self.args.hybrid_training)
        loss = outputs.loss
        logits = {'q_reps': outputs.q_reps,
                  'p_reps': outputs.p_reps,
                  'q_logits': outputs.q_logits,
                  'p_logits': outputs.p_logits}
        return (loss, logits) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
            loss = loss.mean().detach()
        return (loss, None, None)

    # def prediction_step(
    #         self,
    #         model: nn.Module,
    #         inputs: Dict[str, Union[torch.Tensor, Any]],
    #         prediction_loss_only: bool,
    #         ignore_keys: Optional[List[str]] = None,
    # ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    #     query, passage, query_words_ids, passage_words_ids = inputs
    #     with torch.no_grad():
    #         with self.compute_loss_context_manager():
    #             outputs = model(query=query,
    #                             passage=passage,
    #                             query_words_ids=query_words_ids,
    #                             passage_words_ids=passage_words_ids)
    #             loss = outputs.loss / self._dist_loss_scale_factor
    #             logits = torch.tensor(
    #                 [outputs.loss_dense / self._dist_loss_scale_factor,
    #                  outputs.loss_sparse / self._dist_loss_scale_factor],
    #                 device=loss.device)
    #     labels = torch.tensor(0, device=loss.device)  # fake label
    #     return loss, logits, labels

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (EncoderModel,)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            raise ValueError(f"Unsupported model class {self.model}")
        else:
            if state_dict is None:
                state_dict = self.model.state_dict()
            prefix = 'encoder.'
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
            self.model.encoder.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
            # lora_state_dict = get_peft_model_state_dict(self.model.encoder, state_dict)
            # torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
            # print(f"Save adapter model at {output_dir}")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
