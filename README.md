# PromptReps
PromptReps: Prompting Large Language Models to Generate Representations for Zero-Shot Document Retrieval, Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin and Guido Zuccon.
## Installation
We recommend using a conda environment to install the required dependencies.
```bash
conda create -n promptreps python=3.10
conda activate promptreps
```
Our code is build on top of the [Tevatron](https://github.com/texttron/tevatron) library. To install the required dependencies, run the following command:
```bash
git clone https://github.com/texttron/tevatron.git

cd tevatron
pip install transformers datasets peft
pip install deepspeed accelerate
pip install faiss-cpu
pip install ranx
pip install nltk
pip install -e .
cd ..
```
We also use [Pyserini](https://github.com/castorini/pyserini/tree/master) to build inverted index for sparse representations and evaluate the results. 
To install it, run the following command:
```bash
conda install -c conda-forge openjdk=21 maven -y
pip install pyserini
```
If you have any issues with the pyserini installation, please follow this [link](https://github.com/castorini/pyserini/blob/master/docs/installation.md).

## Examples
In this example, we show an experiments with nfcorpus dataset using the Phi-3-mini-4k-instruct model. 
### Step 0: Setup the environment variables.

```bash
BASE_MODEL=microsoft/Phi-3-mini-4k-instruct
DATASET=nfcorpus
OUTPUT_DIR=outputs/${BASE_MODEL}/
```
You can change experiments with other LLMs on huggingface model hub by changing the `BASE_MODEL` variable. 
But you may also need to add prompts in `prompts/${BASE_MODEL}` directory.

Similarly, you can change the dataset by changing the `DATASET` variable to other BEIR dataset names listed [here](https://github.com/beir-cellar/beir?tab=readme-ov-file#beers-available-datasets).

We store the results and intermediate files in the `OUTPUT_DIR` directory.

### Step 1: Encode dense and sparse representation for the queries and documents.

#### Encode queries:
```bash
python encode.py \
    --output_dir=temp \
    --model_name_or_path ${BASE_MODEL} \
    --tokenizer_name ${BASE_MODEL} \
    --per_device_eval_batch_size 64 \
    --query_max_len 512 \
    --normalize \
    --dataset_name Tevatron/beir \
    --dataset_config ${DATASET} \
    --dataset_split test \
    --dense_output_dir ${OUTPUT_DIR}/beir/${DATASET}/dense \
    --sparse_output_dir ${OUTPUT_DIR}/beir/${DATASET}/sparse \
    --encode_is_query \
    --bf16 \
    --query_prefix prompts/${BASE_MODEL}/query_prefix.txt \
    --query_suffix prompts/${BASE_MODEL}/query_suffix.txt \
    --cache_dir cache_models \
    --dataset_cache_dir cache_datasets \
    --sparse_exact_match
```
#### Encode documents:
For large corpus, we sharding the document collection and encode each shard in parallel with multiple GPUs.

For example, if you have two GPUs:
```bash
NUM_AVAILABLE_GPUS=2
for i in $(seq 0 $((NUM_AVAILABLE_GPUS-1)))
do
CUDA_VISIBLE_DEVICES=${i} python encode.py \
        --output_dir=temp \
        --model_name_or_path ${BASE_MODEL} \
        --tokenizer_name ${BASE_MODEL} \
        --per_device_eval_batch_size 64 \
        --passage_max_len 512 \
        --normalize \
        --bf16 \
        --dataset_name Tevatron/beir-corpus \
        --dataset_config ${DATASET} \
        --dense_output_dir ${OUTPUT_DIR}/beir/${DATASET}/dense \
        --sparse_output_dir ${OUTPUT_DIR}/beir/${DATASET}/sparse \
        --passage_prefix prompts/${BASE_MODEL}/passage_prefix.txt \
        --passage_suffix prompts/${BASE_MODEL}/passage_suffix.txt \
        --cache_dir cache_models \
        --dataset_cache_dir cache_datasets \
        --dataset_number_of_shards ${NUM_AVAILABLE_GPUS} \
        --dataset_shard_index ${i} \
        --sparse_exact_match &
done
wait
```

### Step 2: Build dense and sparse indices and retrieve results.

#### Dense retrieval:
```bash
mkdir -p ${OUTPUT_DIR}/beir/${DATASET}/results
python -m tevatron.retriever.driver.search \
    --query_reps ${OUTPUT_DIR}/beir/${DATASET}/dense/query.pkl \
    --passage_reps ${OUTPUT_DIR}'/beir/'${DATASET}'/dense/corpus_*.pkl' \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to ${OUTPUT_DIR}/beir/${DATASET}/results/rank.dense.txt

# convert to trec run format
python -m tevatron.utils.format.convert_result_to_trec --input ${OUTPUT_DIR}/beir/${DATASET}/results/rank.dense.txt \
                                                       --output ${OUTPUT_DIR}/beir/${DATASET}/results/rank.dense.trec \
                                                       --remove_query
```

#### Sparse retrieval:

```bash
# Build inverted index
python -m pyserini.index.lucene \
    --collection JsonVectorCollection \
    --input ${OUTPUT_DIR}/beir/${DATASET}/sparse \
    --index ${OUTPUT_DIR}/beir/${DATASET}/sparse/index \
    --generator DefaultLuceneDocumentGenerator \
    --threads 16 \
    --impact --pretokenized

# search
python -m pyserini.search.lucene \
    --index ${OUTPUT_DIR}/beir/${DATASET}/sparse/index \
    --topics ${OUTPUT_DIR}/beir/${DATASET}/sparse/query.tsv \
    --output ${OUTPUT_DIR}/beir/${DATASET}/results/rank.sparse.trec \
    --output-format trec \
    --batch 32 --threads 16 \
    --hits 1000 \
    --impact --pretokenized --remove-query
```

#### Hybrid the dense and sparse rankings:
```bash
python hybrid.py \
--run_1 ${OUTPUT_DIR}/beir/${DATASET}/results/rank.dense.trec \
--run_2 ${OUTPUT_DIR}/beir/${DATASET}/results/rank.sparse.trec \
--alpha 0.5 \
--save_path ${OUTPUT_DIR}/beir/${DATASET}/results/rank.hybrid.trec
```


### Step 3: Evaluate the results:

```bash
# Dense results
python -m pyserini.eval.trec_eval -c -m recall.100,1000 -m ndcg_cut.10 beir-v1.0.0-${DATASET}-test  ${OUTPUT_DIR}/beir/${DATASET}/results/rank.dense.trec

#Sparse results
python -m pyserini.eval.trec_eval -c -m recall.100,1000 -m ndcg_cut.10 beir-v1.0.0-${DATASET}-test ${OUTPUT_DIR}/beir/${DATASET}/results/rank.sparse.trec

#Hybrid results
python -m pyserini.eval.trec_eval -c -m recall.100,1000 -m ndcg_cut.10 beir-v1.0.0-${DATASET}-test ${OUTPUT_DIR}/beir/${DATASET}/results/rank.hybrid.trec
```

You will get following reults:
```
Dense results:
recall_100              all     0.2617
recall_1000             all     0.5531
ndcg_cut_10             all     0.2780

Sparse results:
recall_100              all     0.2410
recall_1000             all     0.4415
ndcg_cut_10             all     0.2938

Hybrid results:
recall_100              all     0.2853
recall_1000             all     0.5678
ndcg_cut_10             all     0.3325

```
