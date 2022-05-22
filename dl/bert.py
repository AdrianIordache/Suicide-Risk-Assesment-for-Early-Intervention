import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import pdb
import json

from utils import metric_evaluation

import wandb

import torch
from transformers import BertTokenizer, BertTokenizerFast, DistilBertTokenizerFast, BertForSequenceClassification, DistilBertForSequenceClassification, BertModel
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, EvalPrediction

import numpy as np
import random

from dataset import SuicideDataset
from utils import set_seed
 
set_seed(1)

LEVEL = 'post'
epochs = 5
valid_strategy = 2
column_used = 'social_prepocessed_text'

wandb.init(
    project="bmnlp-project",
    name=f"{LEVEL}_level-valid_strategy_{valid_strategy}-column_used_{column_used}",
    entity="andreig"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

verbose = True

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
# max sequence length for each document/sentence sample
max_length = 512

def make_compute_metrics(valid_users):
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
    
        # results = np.array((valid_users, np.squeeze(labels), preds), dtype=object).T
        results = np.zeros((len(valid_users), 3))
        results[:, 0] = valid_users
        results[:, 1] = np.squeeze(labels)
        results[:, 2] = preds

        accuracy, precision, recall, ord_error = metric_evaluation(results, level=LEVEL, verbose=verbose)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'ord_error': ord_error,
        }

    return compute_metrics

if __name__=='__main__':
    dataset = pd.read_csv(f"../data/suicide_{LEVEL}_preprocessed.csv")
    print(dataset.head())

    os.makedirs('metrics_results', exist_ok=True)

    columns = ['user', 'text', 'label', 'prepocessed_text', 'social_prepocessed_text']
    oof_users, oof_labels, oof_predictions = [], [], []
    for fold in range(5):
        X_train = dataset[dataset[f'{valid_strategy}_fold'] != fold][columns]
        y_train = dataset[dataset[f'{valid_strategy}_fold'] != fold]['label'].values

        X_valid = dataset[dataset[f'{valid_strategy}_fold'] == fold][columns]
        y_valid = dataset[dataset[f'{valid_strategy}_fold'] == fold]['label'].values

        print("Label Distribution: {} => {}".format(*np.unique(y_valid, return_counts = True)))
        print(f"Train Samples: {X_train.shape[0]}, Valid Sample: {X_valid.shape[0]}")
        print(f"Train Users: {np.unique(X_train['user'])[:18]}")
        print(f"Valid Users: {np.unique(X_valid['user'])[:18]}")
        print(X_train.head())
        print()
        

        sample_size = 200
        train_texts = list(X_train[column_used].values)
        val_texts = list(X_valid[column_used].values)

        train_labels = list(X_train.label.values)
        valid_labels = list(X_valid.label.values)

        valid_users = list(X_valid.user.values)

        # For DistilBERT:
        model_class, tokenizer_class, pretrained_weights = (DistilBertForSequenceClassification, DistilBertTokenizerFast, 'distilbert-base-uncased')

        ## Want BERT instead of distilBERT? Uncomment the following line:
        #model_class, tokenizer_class, pretrained_weights = (BertForSequenceClassification, BertTokenizerFast, 'bert-base-uncased')

        target_names = set(train_labels)
        # Load pretrained model/tokenizer
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights, num_labels=len(target_names)).to(device)

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        valid_encodings = tokenizer(val_texts, truncation=True, padding=True)

        # convert our tokenized data into a torch Dataset
        train_dataset = SuicideDataset(train_encodings, train_labels)
        valid_dataset = SuicideDataset(valid_encodings, valid_labels)

        # metric = compute_metrics(EvalPrediction(inputs=X_valid.user.values))
        metric = make_compute_metrics(valid_users)

        if LEVEL == 'user':
            logging_steps = 1000
            save_steps = 1000
            eval_steps = 1000
        else:
            logging_steps = 1000
            save_steps = 1000
            eval_steps = 1000

        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=epochs,              # total number of training epochs
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=2,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
            # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
            metric_for_best_model='accuracy',
            greater_is_better=True,
            logging_steps=logging_steps,               # log & save weights each logging_steps
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",     # evaluate each `logging_steps`
            report_to="wandb",
            run_name=f"{LEVEL}_level-valid_strategy_{valid_strategy}-column_used_{column_used}-fold_{fold + 1}"  # name of the W&B run (optional)
        )

        trainer = Trainer(
            model=model,                         # the instantiated Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=valid_dataset,          # evaluation dataset
            compute_metrics=metric,     # the callback that computes metrics of interest
            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # train the model
        trainer.train()

        # evaluate the current model after training
        trainer.evaluate()

        results_valid = trainer.predict(valid_dataset)
        y_labels = results_valid.label_ids
        y_preds = results_valid.predictions.argmax(-1)
        oof_users.extend(valid_users)
        oof_labels.extend(np.squeeze(y_labels))
        oof_predictions.extend(y_preds)

        fold_results = np.zeros((len(valid_users), 3))
        fold_results[:, 0] = valid_users
        fold_results[:, 1] = np.squeeze(y_labels)
        fold_results[:, 2] = y_preds

        accuracy, precision, recall, ord_error = metric_evaluation(fold_results, level=LEVEL, verbose=verbose)

        results_metrics_path = os.path.join('metrics_results', f'json_{LEVEL}_level-valid_strategy_{valid_strategy}-column_used_{column_used}-epochs_{epochs}-fold_{fold + 1}')
        results_json = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'ord_error': ord_error,
            'level': LEVEL,
            'valid_strategy': valid_strategy,
            'column_used': column_used,
            'epochs': epochs,
            'pretrained_weights': pretrained_weights
        }

        # Directly from dictionary
        with open(f'{results_metrics_path}.json', 'w') as outfile:
            json.dump(results_json, outfile)

                
    all_results = np.zeros((len(oof_users), 3))
    all_results[:, 0] = oof_users
    all_results[:, 1] = oof_labels
    all_results[:, 2] = oof_predictions

    accuracy, precision, recall, ord_error = metric_evaluation(all_results, level=LEVEL, verbose=verbose)

    results_metrics_path = os.path.join('metrics_results', f'json_{LEVEL}_level-valid_strategy_{valid_strategy}-column_used_{column_used}-epochs_{epochs}-final')
    results_json = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'ord_error': ord_error,
        'level': LEVEL,
        'valid_strategy': valid_strategy,
        'column_used': column_used,
        'epochs': epochs,
    }

    # Directly from dictionary
    with open(f'{results_metrics_path}.json', 'w') as outfile:
        json.dump(results_json, outfile)