"""
Here we identify the class InputBuilder as a utility class to prepare the input of RoBERTa model from STS benchmark
We also identify a wrapper for the STS benchmark.
"""
import torch
import pandas as pd
import csv


class InputBuilder:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def truncate_pair_of_tokens(self, tokens_a, tokens_b):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= self.max_len - 3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def build_features(self, example):
        tokens_a = self.tokenizer.tokenize(example["sentence1"])
        tokens_b = self.tokenizer.tokenize(example["sentence2"])
        self.truncate_pair_of_tokens(tokens_a, tokens_b)
        tokens = []
        # tokens.append("[CLS]")
        tokens.append(self.tokenizer.cls_token)
        for token in tokens_a:
            tokens.append(token)
        # tokens.append("[SEP]")
        tokens.append(self.tokenizer.sep_token)
        for token in tokens_b:
            tokens.append(token)
        # tokens.append("[SEP]")
        tokens.append(self.tokenizer.sep_token)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_len:
            input_ids.append(0)
            input_mask.append(0)

        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        return input_ids, input_mask


class STSBenchmark(torch.utils.data.Dataset):
    def __init__(self, dataset_path, max_len, tokenizer):
        self.max_len = max_len
        self.dataset_path = dataset_path
        self.dataset = self.read_data()
        self.inputBuilder = InputBuilder(tokenizer, max_len)

    def read_data(self):
        raw_data = pd.read_csv(self.dataset_path, sep="\t", names=list('1234567'), quoting=csv.QUOTE_NONE)
        data = [{"sentence1": raw_data['6'][i],
                 "sentence2": raw_data['7'][i],
                 "similarity": raw_data['5'][i] / 5} for i in range(len(raw_data))]
        return data

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids, input_mask = self.inputBuilder.build_features(example)
        similarity = torch.tensor(example["similarity"], dtype=torch.float32)
        return input_ids, input_mask, similarity.unsqueeze(0)

    def __len__(self):
        return len(self.dataset)
