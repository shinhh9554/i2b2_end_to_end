import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torchcrf import CRF


class BertCrfForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)  # add hidden states and attention if they are here

        loss = None
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            loss = -1 * loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, logits, attention_mask):
        return self.crf.decode(logits, attention_mask.byte())


class BertBiGRUForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBiGRUForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_gru = nn.GRU(config.hidden_size,
                             config.hidden_size // 2,
                             num_layers=2,
                             dropout=classifier_dropout,
                             batch_first=True,
                             bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_gru(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class BertBiGRUFCRForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBiGRUFCRForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        classifier_dropout = 0.7
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_gru = nn.GRU(config.hidden_size,
                             config.hidden_size // 2,
                             num_layers=2,
                             dropout=classifier_dropout,
                             batch_first=True,
                             bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_gru(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            loss = -1 * loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, logits, attention_mask):
        return self.crf.decode(logits, attention_mask.byte())


class BertBiLSTMForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBiLSTMForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_lstm = nn.LSTM(config.hidden_size,
                               config.hidden_size // 2,
                               num_layers=2,
                               dropout=classifier_dropout,
                               batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


class BertBiLSTMFCRForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBiLSTMFCRForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        classifier_dropout = config.hidden_dropout_prob
        self.dropout = nn.Dropout(classifier_dropout)
        self.bi_lstm = nn.LSTM(config.hidden_size,
                               config.hidden_size // 2,
                               num_layers=2,
                               dropout=classifier_dropout,
                               batch_first=True,
                               bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output, hc = self.bi_lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask.byte())
            loss = -1 * loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, logits, attention_mask):
        return self.crf.decode(logits, attention_mask.byte())

