import os
import re
import ast
import math
import copy
import time
import nltk
import string
import pickle
import random
import numpy as np
import pandas as pd
import contractions
import text2emotion as te
import matplotlib.pyplot as plt

from contextlib import redirect_stdout
from bayes_opt import BayesianOptimization
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from typing import Tuple, List, Dict
from lightgbm import LGBMClassifier
from IPython.display import display

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
try:
	from sklearn.utils.testing import ignore_warnings
except:
	from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize

import warnings
warnings.filterwarnings("ignore")

SEED = 42

STOP = set(stopwords.words('english'))

MAP_REVERSE = {
	1: "Supportive", 
	2: "Indicator", 
	3: "Ideation", 
	4: "Behavior", 
	5: "Attempt"
}

OUTPUT = {
	"post_accuracy":  "NA",
	"post_precision": "NA",
	"post_recall":    "NA",
	"post_error":     "NA", 

	"user_accuracy":  "NA",
	"user_precision": "NA",
	"user_recall":    "NA",
	"user_error":     "NA", 

}

class GlobalLogger:
    def __init__(self, path_to_global_logger: str, save_to_log: bool):
        self.save_to_log = save_to_log
        self.path_to_global_logger = path_to_global_logger

        if os.path.exists(self.path_to_global_logger):
            self.logger = pd.read_csv(self.path_to_global_logger)

    def append(self, config_file: Dict, output_file: Dict):
        if self.save_to_log == False: return

        if os.path.exists(self.path_to_global_logger) == False:
            config_columns = [key for key in config_file.keys()]
            output_columns = [key for key in output_file.keys()]

            columns = config_columns + output_columns 
            logger = pd.DataFrame(columns = columns)
            logger.to_csv(self.path_to_global_logger, index = False)
            
        self.logger = pd.read_csv(self.path_to_global_logger)
        sample = {**config_file, **output_file}
        columns = [key for (key, value) in sample.items()]

        row = [value for (key, value) in sample.items()]
        row = np.array(row)
        row = np.expand_dims(row, axis = 0)

        sample = pd.DataFrame(row, columns = columns)
        self.logger = self.logger.append(sample, ignore_index = True)
        self.logger.to_csv(self.path_to_global_logger, index = False)

    
    def get_version_id(self):
        if os.path.exists(self.path_to_global_logger) == False: return 0
        logger = pd.read_csv(self.path_to_global_logger)
        ids = logger["id"].values
        if len(ids) == 0: return 0
        return ids[-1] + 1
    
    def view(self):
        from IPython.display import display
        display(self.logger)


class Logger:
    def __init__(self, path_to_logger: str = 'logger.log', distributed = False):
        from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

        if distributed == False:
            handler1 = StreamHandler()
            handler1.setFormatter(Formatter("%(message)s"))
            self.logger.addHandler(handler1)

        handler2 = FileHandler(filename = path_to_logger)
        handler2.setFormatter(Formatter("%(message)s"))
        self.logger.addHandler(handler2)

    def print(self, message):
        self.logger.info(message)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

##### 8305 samples
##### No other information
##### https://www.kaggle.com/hyunkic/twitter-depression-dataset
def twitter_depression_dataset(source: int = 0) -> pd.DataFrame:
	depression_tweets = pd.read_csv("data/depression-tweets/d_tweets.csv")
	depression_tweets = depression_tweets[["tweet"]]
	depression_tweets["label"] = "depression"

	nondepression_tweets = pd.read_csv("data/depression-tweets/non_d_tweets.csv")
	nondepression_tweets = nondepression_tweets[["tweet"]]
	nondepression_tweets["label"] = "non-depressed"
	
	dataset = pd.concat([depression_tweets, nondepression_tweets], axis = 0)
	dataset = dataset.sample(frac = 1, random_state = SEED).reset_index(drop = True)
	dataset = dataset.rename(columns = {"tweet" : "text"})
	dataset["source"] = source
	return dataset

##### 20363 samples
##### No other information
##### https://www.kaggle.com/xavrig/reddit-dataset-rdepression-and-rsuicidewatch
def reddit_depression_suicidewatch(source: int = 0) -> pd.DataFrame:
	dataset = pd.read_csv("data/depression-suicide-reddit/reddit_depression_suicidewatch.csv")
	dataset['label']  = dataset['label'].map({"SuicideWatch" : "suicide", "depression" : "depression"})
	dataset['source'] = source 
	return dataset


##### 348124 samples
##### "SuicideWatch" from Dec 16, 2008, to Jan 2, 2021 
##### "depression"   from Jan 1, 2009, to Jan 2, 2021
##### https://www.kaggle.com/nikhileswarkomati/suicide-watch
def reddit_depression_suicide_teens(source: int = 0) -> pd.DataFrame:
	dataset = pd.read_csv("data/depression-suicide-reddit/suicide_depression_teens.csv")
	dataset = dataset.rename(columns = {"class" : "label"})
	dataset = dataset.replace([np.inf, -np.inf], np.nan)
	dataset = dataset.dropna()
	
	dataset['label']  = dataset['label'].map({"SuicideWatch" : "suicide", "depression" : "depression", "teenagers" : "teens"})
	dataset['text']   = dataset['text'].apply(lambda x: np.str_(x))
	dataset['source'] = source 
	return dataset

#### https://github.com/AmanuelF/Suicide-Risk-Assessment-using-Reddit
def suicide_risk_assessment(source: int = 0) -> pd.DataFrame:
	dataset = pd.read_csv("data/suicide-risk-assessment/users_posts_labels.csv")

	dataset["n_char"] = dataset["Post"].apply(lambda x: len(x))

	complete_users   = copy.deepcopy(dataset[dataset['n_char'] != 32759])
	incomplete_users = copy.deepcopy(dataset[dataset['n_char'] == 32759])

	# user = dataset[dataset['User'] == 'user-485']
	# text = user.Post.values[0]
	# print(repr(text))
	# print("\n\n\n\n\n")
	# print(text.replace("\\r\\r", " "))

	complete_users['Post'] = complete_users['Post'].apply(lambda x: ast.literal_eval(x.replace("\\r\\r", " "))) #x.strip("][").replace("', '", ' #### ')[1 : -1].split(" #### "))
	complete_users = complete_users.explode("Post").reset_index(drop = True)
	complete_users = complete_users.rename(columns = {"User" : "user", "Label" : "label", "Post" : "text"})

	incomplete_users['Post'] = incomplete_users['Post'].apply(lambda x: ast.literal_eval(x.replace("\\r\\r", " ") + "']")[:-1])
	incomplete_users = incomplete_users.explode("Post").reset_index(drop = True)
	incomplete_users = incomplete_users.rename(columns = {"User" : "user", "Label" : "label", "Post" : "text"})

	dataset = pd.concat([complete_users, incomplete_users], axis = 0)
	dataset = dataset.sample(frac = 1, random_state = SEED).reset_index(drop = True)
	dataset = dataset.drop(["n_char"], axis = 1, inplace = False)

	dataset['user'] = dataset['user'].apply(lambda x: int(x.split("-")[-1]))
	if source is not None: dataset['source'] = source 
	return dataset

def metric_evaluation(results: np.array, level: str = "post", verbose: bool = False) -> Tuple:
	"""
	Metric Evaluation function, returns accuracy, precision, recall, ordinal error (check paper) at user or post level
	
	:param results: numpy array of shape (test_size, 3), results[:, 0] -> user_id, results[:, 1] -> labels, results[:, 2] -> prediction,
	:param level  : 'post' or 'user', if 'user' we group predictions by user_id and select the most voted prediction
	:param verbose: if True display metrics before return 
	:return: (accuracy, precision, recall, ordinal_error) 
	"""

	RD = lambda x: np.round(x, 3)
	assert level in ["post", "user"], "Level should be in ['post', 'user']"

	if level == "user": 
		results = pd.DataFrame(results, columns = ['user', 'labels', 'predictions'])
		users, labels, predictions = [], [], []
		for group_idx, group in results.groupby("user"):
			preds = group.values[:, 2]
			values, counts = np.unique(preds, return_counts = True)
			
			user = group.values[:, 0][0]
			label = group.values[:, 1][0]
			prediction = values[np.argmax(counts)]
			
			users.append(user)
			labels.append(label)
			predictions.append(prediction)

		labels      = np.array(labels)
		predictions = np.array(predictions)
		
		tp = sum(labels == predictions)
		fp = sum(labels  < predictions)
		fn = sum(labels  > predictions)
		oe = sum((labels - predictions) > 1)

		accuracy  = RD(tp / labels.shape[0])
		ord_error = RD(oe / labels.shape[0])
		precision = RD(tp / (tp + fp))
		recall    = RD(tp / (tp + fn))

	else:
		tp = sum(results[:, 1] == results[:, 2])
		fp = sum(results[:, 1]  < results[:, 2])
		fn = sum(results[:, 1]  > results[:, 2])
		oe = sum((results[:, 1] - results[:, 2]) > 1)

		accuracy  = RD(tp / results.shape[0])
		ord_error = RD(oe / results.shape[0])
		precision = RD(tp / (tp + fp))
		recall    = RD(tp / (tp + fn))

	if verbose: 
		print(f"[Level: {level}] Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, Ordinal Error: {ord_error}")
		# print("Confusion Matrix")
		# if level == "user":
		# 	print(confusion_matrix(labels, predictions))
		# else:
		# 	print(confusion_matrix(results[:, 1], results[:, 2]))

	return (accuracy, precision, recall, ord_error)


def check_samples(dataset: pd.DataFrame, samples: int = 50, draw: int = 150) -> None:
	for idx in dataset.sample(samples).index:
		print("=" * draw)
		print(dataset['text'].iloc[idx])
		print("\n\n")
		print(dataset['prepocessed_text'].iloc[idx])
		print("\n\n")
		print(dataset['social_prepocessed_text'].iloc[idx])
		print(f"Label: {MAP_REVERSE[dataset['label'].iloc[idx]]}")
		# print(f"Emotion: {te.get_emotion(dataset['text'].iloc[idx])}")
		print("=" * draw)
		print('\n\n\n\n')

def generate_users_dataset(posts_dataset: pd.DataFrame) -> pd.DataFrame:
	samples = pd.DataFrame(columns = dataset.columns.tolist())
	for idx, data in dataset.groupby("user"):
		user   = data['user'].values[0]
		text   = " ".join([str(elem) for elem in data['text'].values.tolist()])
		label  = data['label'].values[0]
		one    = data['only_one'].values[0]
		fold_1 = data['1_fold'].values[0]
		fold_2 = data['2_fold'].values[0]

		preprocessed_text        = " ".join([str(elem) for elem in data['prepocessed_text'].values.tolist()])
		social_preprocessed_text = " ".join([str(elem) for elem in data['social_prepocessed_text'].values.tolist()])

		sample = [user, text, label, one, fold_1, fold_2, preprocessed_text, social_preprocessed_text]
		samples.loc[len(samples)] = sample
		# break

	emotions = ['happy', 'angry', 'surprise', 'sad', 'fear']
	samples[emotions] = samples['text'].apply(lambda x: pd.Series(te.get_emotion(x)))
	samples.to_csv("data/suicide_users_preprocessed.csv", index = False)

if __name__ == "__main__":
	pass

	# dataset = reddit_depression_suicidewatch(source  = 2)
	# dataset = reddit_depression_suicide_teens(source = 3)
	# display(dataset)