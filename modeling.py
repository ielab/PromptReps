import logging
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from tevatron.retriever.modeling import EncoderModel
from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments as TrainingArguments
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores_dense: Optional[Tensor] = None
    scores_sparse: Optional[Tensor] = None
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    q_logits: Optional[Tensor] = None
    p_logits: Optional[Tensor] = None


class PromptRepsLLM(EncoderModel):
    TRANSFORMER_CLS = AutoModelForCausalLM

    def __init__(self,
                 encoder,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 tokenizer=None,
                 num_pooled_tokens: int = 0,
                 multi_reps=False,
                 word_level_reps: bool = False
                 ):
        super().__init__(encoder, pooling, normalize, temperature)
        self.tokenizer = tokenizer
        self.num_pooled_tokens = num_pooled_tokens
        self.multi_reps = multi_reps
        self.word_level_reps = word_level_reps

    @classmethod
    def load(cls,
             model_name_or_path: str,
             pooling: str = 'cls',
             normalize: bool = False,
             lora_name_or_path: str = None,
             num_pooled_tokens: int = 0,
             multi_reps: bool = False,
             word_level_reps: bool = False,
             **hf_kwargs):
        base_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path,
                                                         device_map='auto',
                                                         **hf_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **hf_kwargs)
        if base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = 0
        if lora_name_or_path:
            print(f"Loading LORA adapter from {lora_name_or_path}")
            lora_config = LoraConfig.from_pretrained(lora_name_or_path, **hf_kwargs)
            lora_model = PeftModel.from_pretrained(base_model, lora_name_or_path, config=lora_config)
            lora_model = lora_model.merge_and_unload()
            model = cls(
                encoder=lora_model,
                pooling=pooling,
                normalize=normalize,
                tokenizer=tokenizer,
                num_pooled_tokens=num_pooled_tokens,
                multi_reps=multi_reps,
                word_level_reps=word_level_reps
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=pooling,
                normalize=normalize,
                tokenizer=tokenizer,
                num_pooled_tokens=num_pooled_tokens,
                multi_reps=multi_reps,
                word_level_reps=word_level_reps
            )
        return model

    def encode_query(self, qry):

        if self.num_pooled_tokens > 0:
            all_logits = [[] for _ in range(qry['input_ids'].shape[0])]
            all_reps = [[] for _ in range(qry['input_ids'].shape[0])]
            append_flags = [True for _ in range(qry['input_ids'].shape[0])]
            generated_tokens = [[] for _ in range(qry['input_ids'].shape[0])]

            input_ids = qry['input_ids']
            attention_mask = qry['attention_mask']
            past_key_values = None

            for _ in range(self.num_pooled_tokens):
                # Mask to filter out sequences that are still active
                active_indices = [i for i, flag in enumerate(append_flags) if flag]

                if not active_indices:
                    break  # If no active sequences, stop the loop

                # Run the encoder only on active sequences
                outputs = self.encoder(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       return_dict=True,
                                       output_hidden_states=True,
                                       use_cache=True,
                                       past_key_values=past_key_values)

                logits = outputs.logits
                past_key_values = outputs.past_key_values

                # Get the logits and hidden states for the last token of active sequences
                next_token_logits = logits[:, -1, :]
                next_token_reps = outputs.hidden_states[-1][:, -1, :]

                # Determine the next tokens
                next_token_ids = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

                if self.multi_reps:
                    next_token_logits = torch.log(1 + torch.relu(next_token_logits))
                    if self.normalize:
                        next_token_reps = torch.nn.functional.normalize(next_token_reps, p=2, dim=-1)

                valid_ids = []
                for idx, active_idx in enumerate(active_indices):
                    token = self.tokenizer.decode(next_token_ids[idx])
                    if '</answer>' in token:  # '"' is our stop generation token
                        append_flags[active_idx] = False

                        if self.word_level_reps:
                            start_word_ids = [0]
                            for i, t in enumerate(generated_tokens[active_idx][1:]):
                                if t[0] == ' ':  # start of a word will be a space
                                    start_word_ids.append(i + 1)

                            grouped_reps = []
                            grouped_logits = []

                            for start, end in zip(start_word_ids, start_word_ids[1:] + [None]):
                                if end is None:
                                    grouped_reps.append(all_reps[active_idx][start:])
                                    grouped_logits.append(all_logits[active_idx][start:])
                                else:
                                    grouped_reps.append(all_reps[active_idx][start:end])
                                    grouped_logits.append(all_logits[active_idx][start:end])

                            grouped_reps = [torch.stack(reps).mean(dim=0) for reps in grouped_reps]
                            grouped_logits = [torch.stack(logits).max(dim=0).values for logits in grouped_logits]

                            all_reps[active_idx] = grouped_reps
                            all_logits[active_idx] = grouped_logits

                    else:
                        all_logits[active_idx].append(next_token_logits[idx])
                        all_reps[active_idx].append(next_token_reps[idx])
                        generated_tokens[active_idx].append(token)
                        valid_ids.append(idx)


                if not valid_ids:
                    break  # If no active sequences, stop the loop

                new_past_key_values = tuple(
                    tuple(k_v[valid_ids, :] for k_v in layer) for layer in past_key_values
                )

                # Update qry with the new input_ids and attention_mask
                new_input_ids = []
                new_attention_mask = []

                for idx in valid_ids:
                    new_input_ids.append(next_token_ids[idx])
                    new_attention_mask.append(torch.cat(
                        [attention_mask[idx], torch.ones([1], device=qry['attention_mask'].device)]))

                input_ids = torch.cat(new_input_ids, dim=0).unsqueeze(1)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                past_key_values = new_past_key_values

            if self.multi_reps:
                return all_logits, all_reps

            # mean pooling
            next_token_reps = torch.stack([torch.stack(reps).mean(dim=0) for reps in all_reps])
            # next_token_logits = torch.stack([torch.stack(logits).mean(dim=0) for logits in all_logits])
            # max pooling
            next_token_logits = torch.stack([torch.stack(logits).max(dim=0).values for logits in all_logits])
            next_token_logits = torch.log(1 + torch.relu(next_token_logits))

        else:
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

    def forward(self,
                query: Dict[str, Tensor] = None,
                passage: Dict[str, Tensor] = None,
                query_words_ids: List[List[int]] = None,
                passage_words_ids: List[List[int]] = None,
                hybrid_training: bool = False, ):

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

        # TODO: Can we avoid for loops here?
        for i in range(len(query_words_ids)):
            mask = torch.zeros_like(q_logits[i])
            mask[query_words_ids[i]] = 1
            q_logits[i] = q_logits[i] * mask
        for i in range(len(passage_words_ids)):
            mask = torch.zeros_like(p_logits[i])
            mask[passage_words_ids[i]] = 1
            p_logits[i] = p_logits[i] * mask

        if self.is_ddp:
            q_reps = self._dist_gather_tensor(q_reps)
            q_logits = self._dist_gather_tensor(q_logits)
            p_reps = self._dist_gather_tensor(p_reps)
            p_logits = self._dist_gather_tensor(p_logits)

        # dense loss
        scores_dense = self.compute_similarity(q_reps, p_reps)
        scores_dense = scores_dense.view(q_reps.size(0), -1)
        target = torch.arange(scores_dense.size(0), device=scores_dense.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))
        loss_dense = self.compute_loss(scores_dense / self.temperature, target)

        if hybrid_training:
            # sparse loss
            scores_sparse = self.compute_similarity(q_logits, p_logits)
            scores_sparse = scores_sparse.view(q_logits.shape[0], -1)
            loss_sparse = self.compute_loss(scores_sparse, target)
            loss = loss_dense + loss_sparse
        else:
            loss = loss_dense

        if self.is_ddp:
            loss = loss * self.world_size  # counter average weight reduction

        return EncoderOutput(
            loss=loss,
            q_reps=q_reps,
            p_reps=p_reps,
            q_logits=q_logits,
            p_logits=p_logits,
        )
