import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

class QaExtract(BertPreTrainedModel):
    def __init__(self, config):
        super(QaExtract, self).__init__(config)
        self.bert = BertModel(config)  #.from_pretrained(bert_pretrain_path)
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)
        self.bert_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.answer_type_classifier = nn.Linear(config.hidden_size, 4)
        self.domain_type_classifier = nn.Linear(config.hidden_size, 2)
        self.answer_softmax = nn.Softmax(-1)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        sequence_output, pooled_output = self.bert(input_ids,
                                                   token_type_ids,
                                                    attention_mask,
                                                    output_all_encoded_layers=output_all_encoded_layers)  # (B, T, 768)
        logits = self.classifier(sequence_output)                                          # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)                                 # ((B, T, 1), (B, T, 1))
        start_logits = start_logits.squeeze(-1)                                            # (B, T)
        end_logits = end_logits.squeeze(-1)                                                # (B, T)
        answer_type_logits = self.answer_type_classifier(pooled_output)
        answer_type_logits = self.answer_softmax(answer_type_logits)

        last_sep = sequence_output[:, -1]
        sep_output = self.activation(self.bert_dense(last_sep))
        domain_type_logits = self.domain_type_classifier(sep_output)

        return start_logits, end_logits, answer_type_logits, domain_type_logits



