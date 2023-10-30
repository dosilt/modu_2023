from transformers import AutoModelForCausalLM, AutoTokenizer
from dataload import load_data, CreateDataset, CollateFn, jsonlload, jsonldump
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from torch.utils.data import DataLoader
from termcolor import colored

from tqdm import tqdm
import json
import torch
import os


def inference(args):
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, args.lora_path).eval()
    model.to('cuda')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
          
    _, _, test_list = load_data(args, 'inference')
    test_set = CreateDataset(args, test_list, tokenizer, 'test')
    collate_fn = CollateFn(tokenizer)
    
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    decoded = []
    with torch.no_grad():
        cnt = 0
        for x in tqdm(test_loader):
            output = model.generate(input_ids=x['input_ids'].to('cuda'),
                                    attention_mask=x['attention_mask'].to('cuda'),
                                    max_length=512,
                                    num_beams=args.num_beams,
                                    early_stopping=True,
                                    eos_token_id=tokenizer.eos_token_id,
                                    use_cache=True,
                                    return_dict_in_generate=True)
            
            for n, sequence in enumerate(output['sequences']):
                prediction = tokenizer.decode(sequence[len(x['input_ids'][0]):], skip_special_tokens=True)
                decoded.append(prediction)
        
    j_list = jsonlload(args.test_path)
    for idx, oup in enumerate(decoded):
        j_list[idx]["output"] = oup

    jsonldump(j_list, f'{args.output_path}/{args.save_name}')
    

if __name__ == '__main__':
    import argparse
    argument = argparse.ArgumentParser()
    argument.add_argument('--model_name', default='beomi/llama-2-ko-7b')
    argument.add_argument('--lora_path', default='results/exp1/lora_weight')
    
    argument.add_argument('--train_path', default='NIKL_SC_2023/nikluge-sc-2023-train.jsonl')
    argument.add_argument('--valid_path', default='NIKL_SC_2023/nikluge-sc-2023-dev.jsonl')
    argument.add_argument('--test_path', default='NIKL_SC_2023/nikluge-sc-2023-test.jsonl')
    
    argument.add_argument('--batch_size', default=4, type=int)
    argument.add_argument('--output_path', default='inference/exp1')
    argument.add_argument('--save_name', default='exp1_inference.jsonl')
    argument.add_argument('--num_beams', default=3, type=int)
    argument.add_argument('--prompt_type', default=2, type=int)
    
    args = argument.parse_args()
    
    args.train_path = f'{os.getcwd()}/{args.train_path}'
    args.valid_path = f'{os.getcwd()}/{args.valid_path}'
    args.test_path = f'{os.getcwd()}/{args.test_path}'
    
    print(colored('------------------------------------------------------------', 'blue'))
    for key, value in vars(args).items():
        print(colored(f'{key:15}', 'green'), end=' : ')
        print(value)
    print(colored('------------------------------------------------------------', 'blue'))
    
    inference(args)