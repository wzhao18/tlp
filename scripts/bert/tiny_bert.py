import torch
from transformers import AutoModel
from torch import nn

class TinyBertModel:
    def __init__(self, self_sup_model) -> None:

        ########### net
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        print('load self_sup_model', self_sup_model)
        if len(self_sup_model) > 0:
            checkpoint = torch.load(self_sup_model)
            model.load_state_dict(checkpoint)


        model.lm_head = nn.Identity()

        self.model = model
        
        
        