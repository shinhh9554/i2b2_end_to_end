from transformers import BertTokenizer, BertConfig, BertForTokenClassification
from bert_modeling import (
    BertCrfForTokenClassification,
    BertBiLSTMForTokenClassification,
    BertBiLSTMFCRForTokenClassification,
    BertBiGRUForTokenClassification,
    BertBiGRUFCRForTokenClassification
)

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "base": BertForTokenClassification,
	"base-crf": BertCrfForTokenClassification,
	"bilstm": BertBiLSTMForTokenClassification,
	"bilstm-crf": BertBiLSTMFCRForTokenClassification,
	"bigru": BertBiGRUForTokenClassification,
	"bigru-crf": BertBiGRUFCRForTokenClassification
}

MODEL_NAME_OR_PATH = {
    "scibert": 'allenai/scibert_scivocab_cased',
	"biobert": 'dmis-lab/biobert-base-cased-v1.1',
    'clinicalbert': "emilyalsentzer/Bio_ClinicalBERT",
	"bert": 'bert-base-cased',
}


