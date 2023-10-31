from torch.utils.data import Dataset
import numpy as np
import torch
import json

## jsonl 파일 불러오는 부분 ##
def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]
    return j_list

## json 파일을 list로 변경 ##
# dataset 만들기 위한 밑작업 #
def json2list(json_data):
    temp = []
    for data in json_data:
        temp.append([data['input']['sentence1'], data['input']['sentence3'], data['output']])
    return temp

def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')
            
            
## 데이터 불러오는 부분 ##
def load_data(args, types='train'):
    train_json = jsonlload(args.train_path)
    valid_json = jsonlload(args.valid_path)
    test_json = jsonlload(args.test_path)
    
    train_list = json2list(train_json)
    valid_list = json2list(valid_json)
    test_list = json2list(test_json)
    if types=='train' and args.add_valid:
        return train_list+valid_list, valid_list, test_list
    return train_list, valid_list, test_list


## Dataset Class 만들기 ##
class CreateDataset(Dataset):
    def __init__(self, args, data, tokenizer):
        super().__init__()
        self.prompt_type = args.prompt_type
        self.data = data
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        s1, s3, s2 = self.data[idx]
        if self.prompt_type == 0:
            prompt = f'문장1과 문장3 사이의 문장2에 올 적절한 문장을 작성하세요.\n문장1:{s1}\n문장2:\n문장3:{s3}\n문장2:'
        elif self.prompt_type == 1:
            prompt = f'문장1:{s1}\n문장2:\n문장3:{s3}\n주어진 문장1과 문장3 사이의 문장2에 올 적절한 문장을 작성하세요.\n문장2:'
        elif self.prompt_type == 2:
            prompt = f'Write an appropriate sentence 2 that goes between the given sentences 1 and 3.\nsentence1:{s1}\nsentence2:\nsentence3:{s3}'
        elif self.prompt_type == 3:
            prompt = f'sentence1:{s1}\nsentence2:\nsentence3:{s3}\nWrite an appropriate sentence 2 that goes between the given sentences 1 and 3.'
        elif self.prompt_type == 4:
            prompt = f'sentence1:{s1}\nsentence2:\nsentence3:{s3}\nRead the given sentences 1 and 3, and create an appropriate sentence 2 between sentences 1 and 3 so that the context continues naturally.'
            
        label = f'{s2}{self.tokenizer.eos_token}'
        return prompt, label
        
        
## Custom Collate function 설계 ##
# 1. seq2seq 방식의 label 설계
# 2. padding setting -> batch 1이라서 영향 없음.
class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, datas):
        prompts, response = [], []

        for input_prompt, label in datas:
            prompts.append(input_prompt+label)
            response.append(label)
        
        input_tokens = self.tokenizer(prompts, padding=True, return_tensors='pt')
        input_labels = [len(x) for x in self.tokenizer(response).input_ids]
        
        input_ids = input_tokens.input_ids
        attention_mask = input_tokens.attention_mask
        labels = []
        for n, i in enumerate(input_ids):
            temp = i.clone()
            temp[:-input_labels[n]] = -100
            labels.append(temp)
        labels = torch.stack(labels)
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
        
if __name__ == '__main__':
    from transformers import AutoTokenizer
    import argparse
    import os
    
    argument = argparse.ArgumentParser()
    argument.add_argument('--train_path', default='NIKL_SC_2023/nikluge-sc-2023-train.jsonl', help='train data 경로')
    argument.add_argument('--valid_path', default='NIKL_SC_2023/nikluge-sc-2023-dev.jsonl', help='dev data 경로')
    argument.add_argument('--test_path', default='NIKL_SC_2023/nikluge-sc-2023-test.jsonl', help='test data 경로')
    argument.add_argument('--model_name', default='beomi/llama-2-ko-7b')
    argument.add_argument('--add_valid', action='store_true')
    argument.add_argument('--prompt_type', default=4)
    args = argument.parse_args()
    
    args.train_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.train_path}'
    args.valid_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.valid_path}'
    args.test_path = f'{os.path.dirname(os.path.realpath(__file__))}/{args.test_path}'
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
    train_list, valid_list, test_list = load_data(args)
    train_set = CreateDataset(args, train_list, tokenizer)
    valid_set = CreateDataset(args, valid_list, tokenizer)
    test_set = CreateDataset(args, test_list, tokenizer, 'test')

    max_train_len, max_valid_len = 0, 0
    for x in train_set:
        max_train_len = max(max_train_len, len(x[0]) + len(x[1]))
    print(max_train_len)
    
    for x in valid_set:
        max_valid_len = max(max_valid_len, len(x[0]) + len(x[1]))
    print(max_valid_len)
    
    
    from torch.utils.data import DataLoader
    collate_fn = CollateFn(tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=2, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=2, collate_fn=collate_fn)
    
    # for x in train_loader:
    #     print(x)
    #     break
    
    for x in test_loader:
        print(x)
        break
