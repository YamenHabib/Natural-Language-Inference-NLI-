"""
TODO: add file description
"""
import torch


class FeatureExtractor:
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
        label = torch.tensor(example["label"], dtype=torch.int64)
        return input_ids, input_mask, label


class GlueDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, tokenizer):
        self.max_len = max_len
        self.dataset = dataset
        self.feature_extractor = FeatureExtractor(tokenizer, max_len)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_ids, input_mask, label = self.feature_extractor.build_features(example)
        return input_ids, input_mask, label

    def __len__(self):
        return len(self.dataset)
