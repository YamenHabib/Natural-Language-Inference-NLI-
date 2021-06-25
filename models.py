import torch
from transformers import RobertaModel

class ROBERTAOnMRPC(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAOnMRPC, self).__init__()
        self.base_model = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 128)
        self.bn1 = torch.nn.LayerNorm(128)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(128, 2)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)

    def forward(self, input_ids, attention_mask):
        _, x = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        out = self.l2(x)
        return out, x

class ROBERTAOnSTS(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAOnSTS, self).__init__()
        self.base_model = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 128)
        self.bn1 = torch.nn.LayerNorm(128)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(128, 1)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.l2.weight)

    def forward(self, input_ids, attention_mask):
        _, x = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        out = self.l2(x)
        return out, x


class ROBERTA_FT_MRPC(torch.nn.Module):
    def __init__(self, pretrained, dropout_rate=0.3):
        super(ROBERTA_FT_MRPC, self).__init__()
        self.base_model = pretrained
        self.l2 = torch.nn.Linear(128, 2)
        self.l3 = torch.nn.Softmax(1)
        torch.nn.init.xavier_uniform_(self.l2.weight)

    def forward(self, input_ids, attention_mask):
        _, x = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = self.l2(x)
        if not self.training:
            x = self.l3(x)
        return x, None  # return None value also for compatibility in train function
