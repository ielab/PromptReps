import logging
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from tevatron.retriever.modeling import EncoderModel
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    loss_dense: Optional[Tensor] = None
    loss_sparse: Optional[Tensor] = None
    q_flops_loss: Optional[Tensor] = None
    p_flops_loss: Optional[Tensor] = None
    scores_dense: Optional[Tensor] = None
    scores_sparse: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None


class PromptRepsLLM(EncoderModel):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def encode_query(self, qry):
        outputs = self.encoder(**qry, return_dict=True, output_hidden_states=True)
        logits = outputs.logits
        sequence_lengths = qry['attention_mask'].sum(dim=1) - 1
        batch_ids = torch.arange(len(qry['input_ids']), device=logits.device)

        # next token logits
        next_token_logits = logits[batch_ids, sequence_lengths]
        next_token_logits = torch.log(1 + torch.relu(next_token_logits))

        # next token hidden states
        next_token_reps = outputs.hidden_states[-1][batch_ids, sequence_lengths]
        if self.normalize:
            next_token_reps = torch.nn.functional.normalize(next_token_reps, p=2, dim=-1)
        return next_token_logits, next_token_reps

    def encode_passage(self, psg):
        # encode passage is the same as encode query
        return self.encode_query(psg)

    @staticmethod
    def _flops(inputs):
        return torch.sum(torch.mean(torch.abs(inputs), dim=0) ** 2)

    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        q_out = self.encode_query(query) if query else None
        p_out = self.encode_passage(passage) if passage else None
        # for inference
        if q_out is None or p_out is None:
            return EncoderOutput(
                q_reps=q_out,
                p_reps=p_out
            )
        q_logits, q_reps = q_out
        p_logits, p_reps = p_out

        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            q_logits = self._dist_gather_tensor(q_logits)
            p_reps = self._dist_gather_tensor(p_reps)
            p_logits = self._dist_gather_tensor(p_logits)


        scores_dense = self.compute_similarity(q_reps, p_reps)
        scores_dense = scores_dense.view(q_reps.size(0), -1)

        target = torch.arange(scores_dense.size(0), device=scores_dense.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss_dense = self.compute_loss(scores_dense / self.temperature, target)

        scores_sparse = self.compute_similarity(q_logits, p_logits)
        scores_sparse = scores_sparse.view(q_logits.size(0), -1)

        loss_sparse = self.compute_loss(scores_sparse, target)

        q_flops_loss = 0.01 * self._flops(q_logits)
        p_flops_loss = 0.01 * self._flops(p_logits)

        loss = loss_dense + loss_sparse + q_flops_loss + p_flops_loss
        if self.is_ddp:
            loss = loss * self.world_size  # counter average weight reduction
        return EncoderOutput(
            loss=loss,
            loss_dense=loss_dense,
            loss_sparse=loss_sparse,
            q_flops_loss=q_flops_loss,
            p_flops_loss=p_flops_loss,
        )
