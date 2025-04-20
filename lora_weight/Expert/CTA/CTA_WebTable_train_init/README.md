---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: Mistral-7B-Instruct-v0.2
model-index:
- name: CTA_WebTable_train_init
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# CTA_WebTable_train_init

This model is a fine-tuned version of [Mistral-7B-Instruct-v0.2](https://huggingface.co/Mistral-7B-Instruct-v0.2) on the CTA_WebTable_train_init dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 512
- total_eval_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 15.0
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.7.1
- Transformers 4.37.2
- Pytorch 2.1.2+cu121
- Datasets 2.16.1
- Tokenizers 0.15.0