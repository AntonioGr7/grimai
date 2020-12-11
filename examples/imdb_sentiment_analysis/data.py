from transformers import AutoTokenizer
import torch

class Data:
    def __init__(self,texts,labels):
        self.class_mapping = {"positive":0,"negative":1}
        self.texts = texts
        self.labels = labels
        self.MAX_LENGTH = 250
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        x = self.texts[item]
        inputs = self.tokenizer.encode_plus(
            x,
            None,
            add_special_tokens=True,
            max_length=self.MAX_LENGTH,
            padding="max_length",
            truncation=True)

        inputs_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        # token_type_ids = inputs['token_type_ids']

        return {
            "text": x,
            "ids": torch.tensor(inputs_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "target": torch.tensor(self.class_mapping[self.labels[item]],dtype=torch.float)
        }