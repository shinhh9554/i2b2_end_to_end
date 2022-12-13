import os
import argparse
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import set_seed, BertConfig, BertTokenizerFast, TrainingArguments, Trainer, DataCollatorForTokenClassification

from metric import Metric
from utils import MODEL_FOR_TOKEN_CLASSIFICATION, MODEL_NAME_OR_PATH

def parse_args():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model_type", default='bigru-crf', type=str)
    parser.add_argument("--model_name_or_path", default='clinicalbert', type=str)

    # Dataset arguments
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--label_all_tokens", default=True, type=bool)
    parser.add_argument("--preprocessing_nu xm_workers", default=1, type=int)

    # Training arguments
    parser.add_argument("--crf", default=True, type=bool)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--do_train", default=True, type=bool)
    parser.add_argument("--do_eval", default=True, type=bool)
    parser.add_argument("--do_predict", default=True, type=bool)
    parser.add_argument("--output_dir", default='output', type=str)
    parser.add_argument("--num_train_epochs", default=30, type=float)
    parser.add_argument("--per_device_train_batch_size", default=300, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=300, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.001, type=float)
    parser.add_argument("--warmup_ratio", default=0.05, type=float)
    parser.add_argument("--fp16", default=True, type=bool)
    parser.add_argument("--evaluation_strategy", default='epoch', type=str)
    parser.add_argument("--logging_steps", default=5, type=int)
    parser.add_argument("--save_strategy", default='epoch', type=str)
    parser.add_argument("--load_best_model_at_end", default=True, type=bool,
                        help="load_best_model_at_end requires the save and eval strategy to match")
    parser.add_argument("--metric_for_best_model", default='f1', type=str)

    args = parser.parse_args()

    return args


def main():
    # Parse the arguments
    args = parse_args()
    now = datetime.now().strftime('%y%m%d_%h%m%s')
    args.output_dir = os.path.join(args.output_dir, f'{args.model_name_or_path}_{args.model_type}_{now}')

    # Set seed
    set_seed(args.seed)

    # Loading Datasets
    raw_datasets = load_dataset('i2b2_end2end')

    # Get unique labels
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    label_list = get_label_list(raw_datasets["train"]['labels'])
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    model_name_or_path = MODEL_NAME_OR_PATH[args.model_name_or_path]
    config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    tokenizer = BertTokenizerFast.from_pretrained(model_name_or_path)
    model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(model_name_or_path, config=config)

    # Set the correspondences label/ID inside the model config
    model.config.label2id = {l: i for i, l in enumerate(label_list)}
    model.config.id2label = {i: l for i, l in enumerate(label_list)}

    b_to_i_label = []
    for idx, label in enumerate(label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in label_list:
            b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    label2id = {l: i for i, l in enumerate(label_list)}
    padding = False
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['words'],
            max_length=args.max_seq_length,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(b_to_i_label[label2id[label[word_idx]]])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batch_size=1,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_raw_datasets['train']
    validation_dataset = processed_raw_datasets['validation']
    test_dataset = processed_raw_datasets['test']

    # Set warmup steps
    total_batch_size = args.per_device_train_batch_size * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * args.num_train_epochs)
    warmup_steps = int(n_total_iterations * args.warmup_ratio)

    # Train & Eval configs
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        evaluation_strategy=args.evaluation_strategy,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        do_train=args.do_train,
        do_eval=args.do_eval
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)

    # Metrics
    metrics = Metric(args, model, label_list)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics,
    )

    # Training
    if args.do_train:
        trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

    # Evaluation
    if args.do_eval:
        metrics = trainer.evaluate(eval_dataset=test_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == '__main__':
    main()