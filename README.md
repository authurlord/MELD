# Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
This repository contains the code for the paper "Efficient Mixture of Experts based on Large Language Models for
Low-Resource Data Preprocessing"

A full version, containing specific task definition, proof of mentioned theorem, and additional experiment result is in [Full Version](./supplementary/LLM_MOE_DB__KDD_Full_Version_v1.pdf)

We provide the checkpoint for our method, as well as the training data for these checkpoints. 

## Data Download
- Due to the large size of the data and checkpoint, we upload all the related checkpoint into a zip file in [link]. Please download the zip file and unzip to the dataset folder.(TBD)
## Requirement

Please check the requirements in [DITTO](https://github.com/megagonlabs/ditto), [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) and [vllm](https://github.com/vllm-project/vllm) for requirements.

## Data Preprocessing Stage

The preprocess stage of baseline data is in `data_preprocess_baseline.ipynb`, the preprocess stage of MELD is in `data_preprocess_MoE.ipynb`.

All data and an end-to-end preprocess pipeline will be released soon.

## RAG Training Stage 

- For RAG model training, the base model is [bge-large-en-1.5](https://huggingface.co/BAAI/bge-large-en-v1.5).
- After transfer and annotate cross-task positive and negative samples in a `.jsonl` file, e.g. `MoE-Example/ER/amazon-google/SBert/amazon-google-blocking-HN.jsonl`, the training command should be like:
```
WANDB_MODE=disabled accelerate launch \
-m FlagEmbedding.baai_general_embedding.finetune.run \
--output_dir MoE-Example/ER/amazon-google/SBert/rerank-em \
--model_name_or_path bge-large-en-1.5 \
--train_data MoE-Example/ER/amazon-google/SBert/amazon-google-blocking-HN.jsonl \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 10 \
--per_device_train_batch_size 16 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 128 \
--passage_max_len 128 \
--train_group_size 2 \
--negatives_cross_device \
--logging_steps 10 \
--query_instruction_for_retrieval "" \
--save_steps 100
```

## Expert Training Stage
- For expert training, the base model is [Mistral-7b-Instruct-0.2 model](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).
- Each data should be registered to `dataset/dataset_info.json`(TBD)
- After registering data, e.g. 'semi-text-w-MoE', the training command should be like:
```
WANDB_MODE=disabled accelerate launch src/train_bash.py     \
--stage sft     \
--model_name_or_path Mistral-7B-Instruct-v0.2    \
--do_train     \
--finetuning_type lora     \
--dataset semi-text-w-MoE     \
--output_dir lora_weight/MoE/ER/Mistral/semi-text-w-MoE \
--overwrite_output_dir     \
--lr_scheduler_type cosine     \
--num_train_epochs 10.0     \
--gradient_accumulation_steps 8     \
--per_device_eval_batch_size 8     \
--fp16     \
--template mistral     \
--lora_r 16 \
--logging_steps 5 \
--plot_loss  \
--lora_target all \
--save_steps 50 \
--use_unsloth \
--quantization_bit 8
```
- If `accelerate` can not successfully initialized, please replace the first line of command into `deepspeed`, as:
```
WANDB_MODE=disabled deepspeed --include localhost:0,1,2,3 src/train_bash.py
```
and change 
```
--gradient_accumulation_steps 1
```

## Router Network $\mathcal{N}$ Training
- The router network training is similar to RAG training.
- The process of training data for $\mathcal{N}$ is in `MoE_query.ipynb`
- The embedding model has two choices. One is [bge-large-en-1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) for short query, and [bge-m3](https://huggingface.co/BAAI/bge-m3) for long query.(TBD)

## LoRA Merge
- The process of LoRA Merge is in `MoE_query.ipynb`
- The code structure is like:
```
peft_model_id = "lora_weight/MoE_CT/add/Mistral/amazon_google-MoE-CT" 
model = PeftModel.from_pretrained(base_model, model_id=peft_model_id,adapter_name='Mistral|amazon_google-MoE-CT') ## Load Base Model and LoRA#1

model.load_adapter("lora_weight/MoE_CT/add/Mistral/amazon_google-MoE-CT", adapter_name="Mistral|amazon_google-MoE-CT") ## LoRA #1
model.load_adapter("lora_weight/MoE_CT/add/Mistral/restaurant-MoE-CT", adapter_name="Mistral|restaurant-MoE-CT") ## LoRA #1

adapter_sequence = ["Mistral|amazon_google-MoE-CT",'Mistral|restaurant-MoE-CT'] ## Here only list 2, you may add more.

model.add_weighted_adapter(
    adapters=adapter_sequence[:2],
    weights=[0.5,0.5],
    adapter_name="Expert-2",
    combination_type="cat"
) ## For simplicity, we equally merge LoRAs, and use cat

model.save_pretrained(model_id = 'merge_experts_semi_text_w',save_directory='lora_weight/MoE_CT/add/Mistral/merge_experts_2/',selected_adapters=[
 'Expert-2']) ## Save the merge expert above
```
- The above LoRA merge only save LoRA weight, and do not require VRAM usage. It can be done with CPU and RAM only.
- Furthur discussion about different merge method can be referred to this [link](https://huggingface.co/docs/peft/main/en/package_reference/lora#peft.LoraModel.add_weighted_adapter) for add_weighted_adapter definition and this [link](https://github.com/huggingface/peft/issues/1155) for discussion of different merge methods.
- For parameter-efficient, `svd` may be the better solutions, however it requires additional computation cost, while `cat` do not need.

## MoE Query

- Please check `vllm_multi_lora_inference.ipynb` for the implement detail. An end-to-end pipeline will be released soon.(TBD)
- The Initialization args are like:
```
def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model="Mistral-7B-Instruct-v0.2",
                             enable_lora=True,
                             max_loras=32, ## 16 for RTX 3090
                             max_lora_rank=64, ## Available for merging 4 LoRAs, that's why we apply top-3 
                             max_cpu_loras=32, 
                             max_num_seqs=256,enforce_eager=True,tensor_parallel_size=8,max_model_len=8192)
    return LLMEngine.from_engine_args(engine_args)
model = initialize_engine()
```
