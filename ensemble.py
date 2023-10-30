from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from collections import Counter
from termcolor import colored
import torch.nn.functional as F
from dataload import jsonlload, jsonldump, json2list
import numpy as np
from tqdm import tqdm
import torch
import os
import json
from torch.nn import CrossEntropyLoss


def crossentropy(model, tokenizer, sample):
    input_tokens = tokenizer(sample, return_tensors='pt', padding=True).to('cuda')
    
    input_ids, attention_mask = input_tokens['input_ids'], input_tokens['attention_mask']

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask).logits
    loss = _crossentropy(input_ids, output, tokenizer)
    return loss

def _crossentropy(labels, logits, tokenizer):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    total = []
    loss_fct = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    for shift_logit, shift_label in zip(shift_logits, shift_labels):
        loss = loss_fct(shift_logit, shift_label)
        total.append(loss.item())
    return total


def score_model_load(args):
    score_models = {}
    for score_model_name in args.score_models:
        model = AutoModelForCausalLM.from_pretrained(score_model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(score_model_name)
        tokenizer.padding_side = 'right'
        score_models[score_model_name] = {'model': model.to('cuda'), 'tokenizer': tokenizer}
    return score_models


def input_sentences(samplelist, idx):
    batching, generate = [], []
    for samples in samplelist:
        sen1, sen3, sen2 = samples[idx]
        batching.append(f'{sen1}\n{sen2}\n{sen3}')
        generate.append(sen2)
    return batching, generate


def ensemble_main(args):
    ## 앙상블할 데이터 불러오는 부분
    samplelist = []
    for path in args.exp_paths:
        sample = jsonlload(path)
        samplelist.append(json2list(sample))
    
    ## 평가 모델 호출 부분
    score_model_dict = score_model_load(args)
    
    sample_len = len(samplelist)
    win_cnt = [0] * sample_len
    decoded = []
    pbar = tqdm(range(len((samplelist[0]))))
    
    for num in pbar:
        batch_result = []
        batching, generate = input_sentences(samplelist, num)
        
        likelihood = []
        for key in score_model_dict.keys():
            model, tokenizer = score_model_dict[key]['model'], score_model_dict[key]['tokenizer']
            scores = crossentropy(model, tokenizer, batching)
            likelihood.append(scores)
        likelihood_score = np.sum(np.array(likelihood), axis=0)
        
        text_count = Counter(generate)
        
        for i in range(len(generate)):
            batch_result.append([generate[i], likelihood_score[i], text_count[generate[i]], i])
            
        batch_result = sorted(batch_result, key=lambda x: [x[1], -x[2], x[3]])
            
        if text_count[generate[0]] >= 3:
            decoded.append(generate[0])
        else:
            decoded.append(batch_result[0][0])
            
    j_list = jsonlload(args.test_path)
    for idx, oup in enumerate(decoded):
        j_list[idx]["output"] = oup

    jsonldump(j_list, f'{args.save_name}')

if __name__ == '__main__':
    import argparse
        
    argument = argparse.ArgumentParser()
    argument.add_argument('--score_models', nargs='+', default=['EleutherAI/polyglot-ko-1.3b', 'EleutherAI/polyglot-ko-5.8b'])
    
    argument.add_argument('--exp_paths', nargs='+', default=['inference/exp1/exp1_inference.jsonl',
                                                             'inference/exp2/exp2_inference.jsonl',
                                                             'inference/exp3/exp3_inference.jsonl',
                                                             'inference/exp4/exp4_inference.jsonl'])
    
    argument.add_argument('--test_path', default='NIKL_SC_2023/nikluge-sc-2023-test.jsonl', help='test data 경로')
    argument.add_argument('--save_name', default='ensemble_v1.jsonl')
    args = argument.parse_args()
    
    args.exp_paths = [f'{os.path.dirname(os.path.realpath(__file__))}/{x}' for x in args.exp_paths]
    args.test_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.test_path}'
    
    print(colored('------------------------------------------------------------', 'blue'))
    for key, value in vars(args).items():
        print(colored(f'{key:15}', 'green'), end=' : ')
        print(value)
    print(colored('------------------------------------------------------------', 'blue'))
    
    ensemble_main(args)    