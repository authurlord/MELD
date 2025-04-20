## Few-Shot Data Selection per task
### Attribute Value Extraction(AVE)
- OA-Mine: from ExtractGPT. [Paper](https://arxiv.org/abs/2310.12537) [Repo](https://github.com/wbsg-uni-mannheim/ExtractGPT)
- Few-shot Selection: we use 20% data from the original repo in `data/processed_datasets/oa-mine/train_0.2.jsonl`, while the full data is `data/processed_datasets/oa-mine/train_1.0.jsonl`. Test data is `data/processed_datasets/oa-mine/test.jsonl`
- Data Augmentation: since original data contains multiple attributes and targed values, e.g. 3 attribute-key values, we extract it as 3 separate instruction-input-output records.
- the processed data is stored in `dataset/AVE`
- the checkpoint data is stored in `lora_weight/Expert/AVE/AVE-oa_mine`
### Data Imputation(DI)
- Restaurant/Walmart/Amazon: from [ER_Magellan benchmarks](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md).
- for dataset `Restaurant`, we merge the left and right table for benchmark `Fodors-Zagats` in ER_Magellan benchmarks, and treat them as one dataset. For dataset `walmart` and `amazon`, we select the left table of `Walmart-Amazon` and the right table of `Amazon-Google` separately. Please check `raw_dataset/DI` for the merged file.
- For the few-shot index for our experiment, we random select: 10% of all data above as few-shot data, 70% of all data as unlabeled data, and the remain 20% data as test. e.g., for dataset `amazon`, please check `raw_dataset/DI/amazon_train_few_index.npy` for the index of labeled file in `raw_dataset/DI/amazon/amazon_all_filter_train.csv`
- Following the setting of baseline paper [IPM](https://github.com/EliasMei/IPM), we set the following attribute as missint:
```
ATTR_DICT = {
    "walmart":["category", "brand"],
    "amazon":["category", "brand"],
    "restaurant":["city"]
}
```
- Data-Augmentation: we run the embedding model $M_{RAG}$ which is fine-tuned over the labeled data, to pseudo-label all the unlabeled data to construct the train file and the in-context demonstration. As result, for dataset `restaurant`, we expand the labeled set from 86 to 566, `amazon` from 2k to 8k, and `walmart` from 242 to 6939.(Such process is not necessary for all dataset, and the file size can be a lot smaller, by filtering representative records via data diversity) Please check the `brand_predict/category_predict` row for the pseudo-label prediction result.
- train/test file: please check `dataset/DI` for the train/test file
- the checkpoint is stored in `lora_weight/Expert/DI`

## Schema Matching

- all data are from paper [SMAT](https://pmc.ncbi.nlm.nih.gov/articles/PMC8487677/pdf/nihms-1722415.pdf) and [repo](https://github.com/JZCS2018/SMAT)
- Since all data for schema matching are highly biased, we do not apply data augmentation and use all file for training. Please check `dataset/SM` for the train/valid file. 
- the checkpoint is stored in `lora_weight/Expert/SM` 

## Relation-Extraction
- all data comes form [TURL_data](https://github.com/sunlab-osu/TURL), the train data is processed from `train.table_rel_extraction.json`, while the test data is processed from `test.table_rel_extraction.json`
- we select 10% of all data as train file. The selection index in stored in `raw_datatset/RE/RE_sample_10_index.npy`
- Similarity, we use $M_{RAG}$ to annotate `top-k` candidate relations and `top-p` most-relative in-context demonstrations, please check `raw_datatset/RE/` for the processed file.
- train/test file: please check `dataset/RE` for the train/test file
- the checkpoint is stored in `lora_weight/Expert/RE` 

## Column Type Annotation
- SimTab and WebTables are from RECA repo.
- We select 20% data of all datasets as training file. For WebTable, we merge `k0,k1,k2,k3` as all train file, and filter 20% of it as few-shot, while we treat `k4` as test; for SimTab, we treat `train_val_hard_jaccard_ranking.jsonl` as train file. All index can be found as `raw_dataset/CTA` correspondingly.
- Similarity, we use $M_{RAG}$ to annotate `top-k` candidate colume type and `top-p` most-relative in-context demonstrations as data augmentation.
- Due to the large size of all data(for single table with multiple column types and rows, we have to treat them as multiple records, with different context), we sample representative context(e.g. subset of each single table) via data diversity. The train/test file is stored in `dataset/CTA`
- the checkpoint is stored in `lora_weight/Expert/CTA`

## Data Cleaning
- All data comes from Baran repo. and the few-shot setting are kept the same with Baran.
- for each dataset, e.g. `hospital`, the sampled index is stored in `raw_dataset/DC/hospital/index.npy`. The clean and dirty data of benchmark is stored in `raw_dataset/DC/hospital/original/clean.csv` and `raw_dataset/DC/hospital/original/dirty.csv`
- Data Augmentation: we use LLM to generate error detection and data cleaning rules via few-shot samples, and use them to augment data. Please check the appendix of this [Paper](https://github.com/SICS-Fundamental-Research-Center/GIDCL/blob/main/supplementary/GIDCL_Revision_v6_appendix.pdf) for the generated rules.
- The test file only correct the error that are detected from the previous Error Detection results. The detection result is stored in `raw_dataset/DC/hospital/detector/detector.npy`, recording all potential error position.
- The train/test file is stored in `dataset/DC`
- the checkpoint is stored in `lora_weight/Expert/DC`

## Entity Matching 
- `Walmart-Amazon`, `Abt-Buy`, `Amazon-Google` dataset are ER benchmark from [ER_Magellan benchmarks](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md). 
- `WDC-All` dataset are benchmar dataset from [WDC Product Corpus](https://webdatacommons.org/largescaleproductcorpus/v2/index.html). We select the small-size dataset. All setting, containing format and train/valid/test split, are kept the same with [ditto](https://github.com/megagonlabs/ditto/tree/master/data/wdc).
- `Semi-text-Watch` and `Semi-Text-Computer` are from repo [PromptEM](https://github.com/ZJU-DAILY/PromptEM).
- For the few-shot setting, please check `raw_dataset/ER/Walmart-Amazon/index.csv` or `index.npy` for our sampled text.
- Data Augmentation: we use offline LLM to extract additional attributes, and replace the original record with generated structural output. The following are LLM-generated attribute per dataset:
```
ATTR_DICT = {
    "Walmart-Amazon":['title', 'category', 'brand', 'modelno', 'price', 'subcategory', 'key_features', 'sku', 'color'],
    "Amazon-Google":['title', 'manufacturer', 'price', 'category', 'subcategory', 'platform', 'edition', 'type', 'modelno'],
    "WDC-All":["title","category","subcategory","brand","modelno","sku","key_features"],
    "Abt-Buy":['name', 'description', 'price', 'category', 'sku', 'brand', 'modelno', 'key_features']
    "Semi-Text-Watch":['title', 'brand', 'color', 'gender', 'sku', 'diameter', 'description'],
    "Semi-Text-Computer":['title', 'category', 'subcategory', 'brand', 'sku', 'type', 'description'],
}
```
We also provide the prompt for the following generation process. Please check `raw_dataset/ER/Walmart-Amazon/enrich_query_walmart_amazon.csv` for an example. We train $M_\text{RAG}$ over few-shot labeled data, to retrieve unlabeled similar pairs for pairwise LLM generation. Please check `raw_dataset/ER/Walmart-Amazon/train.csv` and `test.csv` for the LLM-generated result.
- $M_\text{RAG}$ is also used to generate pseudo-labeled negative pairs. For dataset `Semi-text-Watch` and `Semi-Text-Computer`, we additional retrieve their `sku` as master data, then self-annotate additional positive(same `sku`) and negative data. 
- The train/test file is stored in `dataset/ER`
- the checkpoint is stored in `lora_weight/Expert/ER`



## Training
- We use llama-factory to conduct SFT stage per expert. You need to modify `src/llamafactory/data/loader.py` as below:
- from datasets import DatasetDict, load_dataset, load_from_disk, Dataset
- import pandas as pd
- import numpy as np
- for function `def _load_single_dataset`, please insert the following code block after the final `else`:
```
    else:
        print(data_path,data_name,data_dir,data_files,dataset_attr.split,model_args.cache_dir,model_args.trust_remote_code)
        if data_args.tokenized_path is not None:
            dataset = load_from_disk(data_args.tokenized_path)
            print('load pre-defined arrow')
        elif data_args.train_file_path is not None:
            data_files = data_args.train_file_path.split(',')
            df = pd.DataFrame()
            for data_file in data_files:   
                df_current = pd.read_json(data_file)
                df_current['ids'] = df_current.index
                df = pd.concat([df,df_current])
            dataset = Dataset.from_pandas(df)
            print('loading from json file')
        else:           
            try:
                df = pd.DataFrame()
                for data_file in data_files:   
                    df_current = pd.read_json(data_file)
                    df = pd.concat([df,df_current])
                dataset = Dataset.from_pandas(df)
                print('loading from pandas')
            except:
                dataset = load_dataset(
                    path=data_path,
                    name=data_name,
                    data_dir=data_dir,
                    data_files=data_files,
                    split=dataset_attr.split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.hf_hub_token,
                    streaming=data_args.streaming,
                    num_proc=data_args.preprocessing_num_workers,
                    trust_remote_code=model_args.trust_remote_code,
                )
        print('load_dataset finished')

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)
```
This can skip huggingface-dataset check, and directly load the `json` file locally

- also modify `src/llamafactory/hparams/data_args.py` with the following args:
```
    train_file_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "json path for training file. "
            )
        },
    )
```

## SFT
- Please check `script/mistral_lora_AVE_oa_mine_init.yaml` as a demonstration script for SFT
- We use llama-factory to conduct SFT stage per expert. use the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3  llamafactory-cli train script/mistral_lora_AVE_oa_mine_init.yaml
```
### Inference:
- Please check `evaluation.ipynb` for the inference code. We use `vllm` to conduct efficient lora-based inference, and the output is in `inference` folder.
- All checkpoint is stored in `lora_weight/Expert` folder
### Evaluation:
- Please check `evaluation.ipynb` for evaluation per dataset.
- `Data Cleaning` task requires previous `Error Detection` result, and the original clean/dirty table. Please change the `detector.npy` and test file correspondingly.
- `Relation Extraction` task requires loading multiple ground truth.