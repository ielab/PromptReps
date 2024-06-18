import numpy as np
import glob
from itertools import chain
from tqdm import tqdm
import faiss
from arguments import PromptRepsDataArguments, PromptRepsSearchArguments
from tevatron.retriever.searcher import FaissFlatSearcher
from pyserini.search.lucene import LuceneImpactSearcher, LuceneSearcher
from pyserini.analysis import JWhiteSpaceAnalyzer
import os
import pickle
import json
from contextlib import nullcontext
from hybrid import fuse, write_trec_run

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from dataset import PromptRepsEncodeDataset, PromptRepsEncodeCollator

from tevatron.retriever.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments
from arguments import PromptRepsDataArguments
from tevatron.retriever.modeling import EncoderOutput
from modeling import PromptRepsLLM
from encode import get_valid_tokens_values, get_filtered_ids
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
stopwords = set(stopwords.words('english') + list(string.punctuation))

import logging
logger = logging.getLogger(__name__)


def pickle_load(path):
    with open(path, 'rb') as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


def search_queries(retriever, q_reps, p_lookup, args):
    if args.batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, args.depth, args.batch_size, args.quiet)
    else:
        all_scores, all_indices = retriever.search(q_reps, args.depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def get_run_dict(batch_ids, batch_scores, batch_rankings, remove_query):
    run_dict = {}
    for qid, scores, rankings in zip(batch_ids, batch_scores, batch_rankings):
        run_dict[qid] = {}
        run_dict[qid]['docs'] = {}
        for score, doc in zip(scores, rankings):
            if remove_query:
                if doc == qid:
                    continue
            run_dict[qid]['docs'][doc] = score
        if len(scores)==0:
            run_dict[qid]['min_score'] = 0
            run_dict[qid]['max_score'] = 0
        else:
            run_dict[qid]['min_score'] = min(scores)
            run_dict[qid]['max_score'] = max(scores)
    return run_dict


def sparse_search(sparse_retriever, batch_topics, batch_ids, search_args):
    results = sparse_retriever.batch_search(batch_topics, batch_ids, search_args.depth,
                                            threads=search_args.threads)
    results = [(id_, results[id_]) for id_ in batch_ids]
    sparse_scores = []
    sparse_rankings = []
    for topic, hits in results:
        scores = []
        ranking = []
        for hit in hits:
            scores.append(hit.score)
            ranking.append(hit.docid)
        sparse_scores.append([hit.score for hit in hits])
        sparse_rankings.append(ranking)
    return sparse_scores, sparse_rankings


def max_pooling_run(run, search_args):
    results = {}
    for qid in run:
        token_scores = []
        all_docids = set()
        for token in run[qid]:
            max_scores = {}
            token_docids = set()
            for ranking in token:
                token_docids.update(ranking[qid]['docs'].keys())
                all_docids.update(ranking[qid]['docs'].keys())
            for docid in token_docids:
                max_score = 0
                for ranking in token:
                    if docid in ranking[qid]['docs']:
                        max_score = max(max_score, ranking[qid]['docs'][docid])
                max_scores[docid] = max_score
            token_scores.append(max_scores)
        # sum, sort and cut to topk
        results[qid] = {}
        results[qid]['docs'] = {}
        for docid in all_docids:
            score = 0
            for token_score in token_scores:
                score += token_score.get(docid, 0)
            results[qid]['docs'][docid] = score
        results[qid]['docs'] = dict(sorted(results[qid]['docs'].items(), key=lambda item: item[1], reverse=True)[:search_args.depth])
        results[qid]['min_score'] = min(results[qid]['docs'].values())
        results[qid]['max_score'] = max(results[qid]['docs'].values())
    return results


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    parser = HfArgumentParser((ModelArguments, PromptRepsDataArguments, PromptRepsSearchArguments, TrainingArguments))
    model_args, data_args, search_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.num_pooled_tokens == 0:
        tokenizer.padding_side = 'right'
    else:
        tokenizer.padding_side = 'left'

    filtered_ids = get_filtered_ids(tokenizer)

    if training_args.bf16:
        torch_dtype = torch.bfloat16
    elif training_args.fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = PromptRepsLLM.load(
        model_args.model_name_or_path,
        pooling=model_args.pooling,
        normalize=model_args.normalize,
        lora_name_or_path=model_args.lora_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        num_pooled_tokens=data_args.num_pooled_tokens,
        multi_reps=data_args.multi_reps,
        word_level_reps=data_args.word_level_reps
    )

    encode_dataset = PromptRepsEncodeDataset(
        data_args=data_args,
    )

    encode_collator = PromptRepsEncodeCollator(
        data_args=data_args,
        tokenizer=tokenizer,
    )

    encode_loader = DataLoader(
        encode_dataset,
        batch_size=search_args.batch_size,
        collate_fn=encode_collator,
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    lookup_indices = []

    vocab_dict = tokenizer.get_vocab()
    vocab_dict = {v: k for k, v in vocab_dict.items()}

    model.eval()

    dense_run = {}
    sparse_run = {}
    fusion_run = {}

    dense_retriever_indices = []
    sparse_retriever_indices = []

    if search_args.passage_reps is not None:
        if data_args.multi_reps:
            dense_retriever_indices = glob.glob(f'{search_args.passage_reps}*')
        else:
            dense_retriever_indices = [search_args.passage_reps]

    if search_args.sparse_index is not None:
        if data_args.multi_reps:
            sparse_retriever_indices = glob.glob(f'{search_args.sparse_index}*')
        else:
            sparse_retriever_indices = [search_args.sparse_index]

    for i in range(max(len(dense_retriever_indices), len(sparse_retriever_indices))):
        dense_retriever = None
        sparse_retriever = None
        if dense_retriever_indices:
            index_files = glob.glob(os.path.join(dense_retriever_indices[i], 'corpus*.pkl'))
            logger.info(f'Pattern match found {len(index_files)} files; loading them into dense index.')

            p_reps_0, p_lookup_0 = pickle_load(index_files[0])
            dense_retriever = FaissFlatSearcher(p_reps_0)

            shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
            if len(index_files) > 1:
                shards = tqdm(shards, desc='Loading shards into index', total=len(index_files))
            look_up = []
            for p_reps, p_lookup in shards:
                dense_retriever.add(p_reps)
                look_up += p_lookup
            if search_args.use_gpu:
                num_gpus = faiss.get_num_gpus()
                if num_gpus == 0:
                    logger.error("No GPU found. Back to CPU.")
                else:
                    logger.info(f"Using {num_gpus} GPU")
                    if num_gpus == 1:
                        co = faiss.GpuClonerOptions()
                        co.useFloat16 = True
                        res = faiss.StandardGpuResources()
                        dense_retriever.index = faiss.index_cpu_to_gpu(res, 0, dense_retriever.index, co)
                    else:
                        co = faiss.GpuMultipleClonerOptions()
                        co.shard = True
                        co.useFloat16 = True
                        dense_retriever.index = faiss.index_cpu_to_all_gpus(dense_retriever.index, co,
                                                                            ngpu=num_gpus)
        if sparse_retriever_indices:
            sparse_retriever = LuceneImpactSearcher(os.path.join(sparse_retriever_indices[i], 'index'), None)
            analyzer = JWhiteSpaceAnalyzer()
            sparse_retriever.set_analyzer(analyzer)


        with torch.no_grad(), torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            for (batch_ids, batch, batch_texts) in tqdm(encode_loader):
                lookup_indices.extend(batch_ids)
                batch = batch.to(training_args.device)
                # batch['qids'] = batch_ids
                model_output: EncoderOutput = model(query=batch)
                q_sparse_reps, q_dense_reps = model_output.q_reps

                if dense_retriever is not None:
                    if isinstance(q_dense_reps, list):
                        for qid, reps in zip(batch_ids, q_dense_reps):
                            reps = torch.stack(reps, dim=0)
                            dense_scores, dense_rankings = search_queries(dense_retriever, reps.cpu().detach().float().numpy(), look_up, search_args)
                            if qid not in dense_run:
                                dense_run[qid] = []
                                for scores, ranking in zip(dense_scores, dense_rankings):
                                    dense_run[qid].append([get_run_dict([qid], [scores], [ranking], search_args.remove_query)])
                            else:
                                for i, (scores, ranking) in enumerate(zip(dense_scores, dense_rankings)):
                                    dense_run[qid][i].append(get_run_dict([qid], [scores], [ranking], search_args.remove_query))

                    else:
                        q_dense_reps = q_dense_reps.cpu().detach().float().numpy()
                        dense_scores, dense_rankings = search_queries(dense_retriever, q_dense_reps, look_up, search_args)
                        dense_run.update(get_run_dict(batch_ids, dense_scores, dense_rankings, search_args.remove_query))

                if sparse_retriever is not None:
                    if isinstance(q_sparse_reps, list):
                        for qid, reps, text in zip(batch_ids, q_sparse_reps, batch_texts):
                            batch_topics = []
                            for logits in reps:
                                tokens, values = get_valid_tokens_values(text, tokenizer, logits, vocab_dict, data_args, filtered_ids)
                                query = ""
                                for token, v in zip(tokens, values):
                                    query += (' ' + token) * v
                                batch_topics.append(query.strip())
                            sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics, [qid] * len(batch_topics), search_args)
                            if qid not in sparse_run:
                                sparse_run[qid] = []
                                for scores, ranking in zip(sparse_scores, sparse_rankings):
                                    sparse_run[qid].append([get_run_dict([qid], [scores], [ranking], search_args.remove_query)])
                            else:
                                for i, (scores, ranking) in enumerate(zip(sparse_scores, sparse_rankings)):
                                    sparse_run[qid][i].append(get_run_dict([qid], [scores], [ranking], search_args.remove_query))

                    else:
                        batch_topics = []
                        for _, logits, text in zip(batch_ids, q_sparse_reps, batch_texts):
                            tokens, values = get_valid_tokens_values(text, tokenizer, logits, vocab_dict, data_args, filtered_ids)
                            query = ""
                            for token, v in zip(tokens, values):
                                query += (' ' + token) * v
                            batch_topics.append(query.strip())
                        sparse_scores, sparse_rankings = sparse_search(sparse_retriever, batch_topics, batch_ids, search_args)
                        sparse_run.update(get_run_dict(batch_ids, sparse_scores, sparse_rankings, search_args.remove_query))
        # delete and clean gpu cache
        if dense_retriever:
            del dense_retriever
            torch.cuda.empty_cache()

    if data_args.multi_reps:
        dense_run = max_pooling_run(dense_run, search_args)
        sparse_run = max_pooling_run(sparse_run, search_args)

    if search_args.passage_reps is not None and search_args.sparse_index is not None:
        fusion_run.update(
            fuse(
                runs=[dense_run, sparse_run],
                weights=[search_args.alpha, (1 - search_args.alpha)]
            )
        )

    # check if the save directory exists
    if not os.path.exists(search_args.save_dir):
        os.makedirs(search_args.save_dir)
    else:
        logger.warning(f"Directory {search_args.save_dir} already exists. Files may be overwritten.")

    if len(dense_run) > 0:
        save_file = os.path.join(search_args.save_dir, 'rank.dense.trec')
        write_trec_run(dense_run, save_file, name='dense')
    if len(sparse_run) > 0:
        save_file = os.path.join(search_args.save_dir, 'rank.sparse.trec')
        write_trec_run(sparse_run, save_file, name='sparse')
    if len(fusion_run) > 0:
        save_file = os.path.join(search_args.save_dir, 'rank.hybrid.trec')
        write_trec_run(fusion_run, save_file, name='hybrid')


if __name__ == '__main__':
    main()
