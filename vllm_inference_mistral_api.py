import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from sklearn.metrics import f1_score
import os
import time
from transformers import AutoTokenizer
from tqdm import tqdm
import yaml
import argparse
import json
import subprocess
def json_to_csv_filename(json_file_path):
    base_name = os.path.basename(json_file_path)
    file_name, _ = os.path.splitext(base_name)
    csv_file_name = file_name + ".csv"
    return csv_file_name

def json_to_csv_filename_GEIL(json_file_path):
    return '_'.join(json_file_path.split('/')[-3:]).replace('.json','.csv')

def json_to_csv_filename_Transfer_ER(json_file_path):
    return json_file_path.split('/')[-1].replace('.json','.csv')

def create_folder_if_not_exists(path):
    """
    Create a folder at the specified path if it does not already exist.

    Parameters:
    path (str): The path where the folder should be created.

    Returns:
    None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at: {path}")
    else:
        print(f"Path already exists: {path}")
def add_suffix_to_filename(file_path, suffix):
    directory, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_{suffix}{ext}"
    new_file_path = os.path.join(directory, new_file_name)
    return new_file_path
def main():
    parser = argparse.ArgumentParser(description="A simple command-line argument parser example.")
    
    parser.add_argument("-test_file", "--file", help="Specify the input file")
    parser.add_argument("--guided_choices", action="store_true", help="whether restrict guided_choices")
    parser.add_argument("-json", "--json", action="store_true", help="Enable verbose mode")
    parser.add_argument("--count", type=int, default=1, help="Specify the count")
    parser.add_argument("--gpu_memory_usage", type=float, default=0.85, help="gpu_memory_usage")
    parser.add_argument("--gpu_num", type=int, default=8, help="gpu_memory_usage")
    parser.add_argument("--max_token", type=int, default=512, help="gpu_memory_usage")
    parser.add_argument("--model_path", type=str, default='../model/Mistral-7B-Instruct-v0.2', help="Model path, default is empty")
    parser.add_argument("--temperature", type=float, default=0, help="temperature")
    parser.add_argument("--top_p", type=float, default=1, help="top_p")
    parser.add_argument("-checkpoint_dir", "--directory", type=str, help="Specify the directory path", default='/data/home/wangys/LLaMA-Factory-main/models/Mixtral-sft')
    args = parser.parse_args()
    print('|'.join(args.directory.split('/')))
    # if args.verbose:
    #     print("Verbose mode enabled")
    
    # if args.file:
    #     print(f"Input file: {args.file}")
    
    # print(f"Count: {args.count}")
    # device = "CUDA_VISIBLE_DEVICES=%s" % str(args.count)
    file_path_list = args.file.split(',')
    output_path = 'lora_weight/{}'.format(args.directory.split('/')[-1])
    if(args.model_path!=''):
        output_path = args.model_path
    elif os.path.exists(output_path):
        output_path = output_path
    
    llm = LLM(model=output_path, 
              tensor_parallel_size=args.gpu_num,dtype="half", 
              enforce_eager=True,
              gpu_memory_utilization = args.gpu_memory_usage,
              enable_lora=True,
              disable_log_stats=True,)
    tokenizer = AutoTokenizer.from_pretrained(output_path)
    if(args.json):
        for file_path in file_path_list:
            result = pd.read_json(file_path)
            
            try:
                text_list = result['instruction'].to_list()
            except:
                try:
                    result.columns = ['instruction','input','output']
                    text_list = result.iloc[:,0].to_list()
                except:
                    text_list = result.iloc[:,0].to_list()
            selection_list = list(result.iloc[:,2].unique())
            selection_list = [s for s in selection_list if s!='']
            if(output_path.lower().__contains__('mistral')):
                text_all = ["[INST] %s [/INST]" % str(a) for a in text_list]
            else: 
                text_all = []
                for prompt in tqdm(text_list):
                    messages = [
                        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    text_all.append(text)
            sampling_params = SamplingParams(temperature=args.temperature, 
                                             top_p=args.top_p,
                                             max_tokens=args.max_token)
            if args.guided_choices:
                outputs = llm.generate(text_all, sampling_params,guided_options_request=dict(guided_choice=selection_list),lora_request = LoRARequest('merge', 1, args.directory))
            else:
                outputs = llm.generate(text_all, sampling_params,lora_request = LoRARequest('merge', 1, args.directory))             
            generation_list = []
            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                generation_list.append(generated_text)
            result['predict'] = generation_list
            print(result['predict'].value_counts())
            def Transfer(row):
                if(row['output'].__contains__('mismatch')) or (row['output'].__contains__('dismatch')):
                    label = 0
                else:
                    label = 1
                if(row['predict'].__contains__('mismatch')) or (row['predict'].__contains__('dismatch')):
                    predict = 0
                else:
                    predict = 1
                return label,predict
            result_output = result.apply(Transfer,axis=1,result_type='expand')
            from sklearn.metrics import precision_score,recall_score,f1_score
            # print(file_path)
            print('Model:{}\n\nFile:{}\n\nPrecision:{}\n\nRecall:{}\n\nF1:{}'.format(args.directory.split('/')[-1],file_path,precision_score(y_true=result_output[0],y_pred=result_output[1]),recall_score(y_true=result_output[0],y_pred=result_output[1]),f1_score(y_true=result_output[0],y_pred=result_output[1])))
            # From Here Run Mistral Baselines
            create_folder_if_not_exists('inference')

            result.to_csv('inference/{}'.format(json_to_csv_filename_Transfer_ER(file_path)))
            print('save result to inference/{}'.format(json_to_csv_filename_Transfer_ER(file_path)))

    else:
        text_list = np.load(args.file)
        prompts = ["A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %s ASSISTANT:" % str(a) for a in text_list] ## Vicuna
        sampling_params = SamplingParams(temperature=0, top_p=1,max_tokens=512,logprobs=1)
        outputs = llm.generate(prompts, sampling_params)
        generation_list = []
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            generation_list.append(generated_text)
        print(add_suffix_to_filename(args.file, "output"))
        np.save(add_suffix_to_filename(args.file, "output"),np.array(generation_list))
    command_del = "rm -rf %s" % output_path
    if(args.model_path==''):
        subprocess.run(command_del, shell=True, capture_output=True, text=True)
if __name__ == "__main__":
    main()
