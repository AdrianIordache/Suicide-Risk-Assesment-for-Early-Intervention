from utils import *
from preprocessing import *

GLOBAL_LOGGER = GlobalLogger(
    path_to_global_logger = f'logs/global_logger.csv',
    save_to_log = False
)

ID = 353
LEVEL = 'post'
SAVE_MODEL = False

CFG = {
	"id": GLOBAL_LOGGER.get_version_id(),
	"model": SVC,
	"parameters": {'C': 1, 'random_state': 42},

	"columns": ['social_prepocessed_text', 'happy', 'angry', 'surprise', 'sad', 'fear'], # ['user', 'text', 'happy', 'angry', 'surprise', 'sad', 'fear'],
	"transformers": [
		 	# {"name" : 'user_id'   , "algorithm": MinMaxScaler   , "parameters": {}                                                                                                               ,  "columns": ['user']},
		 	{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 1)},  "columns": 'social_prepocessed_text'}, #, "token_pattern": r'(?u)\b\w\w+\b|!|,|.|\?|\"|\''}
	],
	"column_transforms": {
		'remainder': StandardScaler(),
		'n_jobs': -1
	},
	"valid_strategy": 1,
	"one_fold": False,
	"observation": "From Bayesian Optimization"
}


dataset = pd.read_csv(f"data/suicide_{LEVEL}_preprocessed.csv")
display(dataset.columns.tolist())

dataset['label']    = dataset['label'].map({"Supportive" : 1, "Indicator" : 2, "Ideation" : 3, "Behavior" : 4, "Attempt" : 5})
dataset['w_counts'] = dataset['prepocessed_text'].apply(lambda x: len(word_tokenize(str(x))))

# plt.figure(figsize = (18, 10))
# plt.bar(*np.unique(dataset['w_counts'], return_counts = True))
# plt.show()

# check_samples(dataset, samples = 100)
dataset['social_prepocessed_text'] = dataset['social_prepocessed_text'].apply(lambda x: " ".join([contractions.fix(word) for word in word_tokenize(x)]))

print(f"Config File: {CFG}")
oof_users, oof_labels, oof_predictions = [], [], [] 
for fold in range(5):
	print("=" * 75 + f" FOLD {fold} " + "=" * 75)
	X_train = dataset[(dataset[f'{CFG["valid_strategy"]}_fold'] != fold) & (dataset['w_counts'] > 0)][CFG['columns']]
	y_train = dataset[(dataset[f'{CFG["valid_strategy"]}_fold'] != fold) & (dataset['w_counts'] > 0)][['user', 'label']].values

	X_valid = dataset[dataset[f'{CFG["valid_strategy"]}_fold'] == fold][CFG['columns']]
	y_valid = dataset[dataset[f'{CFG["valid_strategy"]}_fold'] == fold][['user', 'label']].values

	# print("Label Distribution: {} => {}".format(*np.unique(y_valid, return_counts = True)))
	# print(f"Train Samples: {X_train.shape[0]}, Valid Sample: {X_valid.shape[0]}")
	# print(f"Train Users: {np.unique(X_train['user'])[:18]}")
	# print(f"Valid Users: {np.unique(X_valid['user'])[:18]}")

	transformers = [(transform_dict['name'], transform_dict['algorithm'](**transform_dict["parameters"]), transform_dict['columns']) \
							for transform_dict in CFG['transformers']]

	ct_parameters = copy.deepcopy(CFG["column_transforms"])
	ct_parameters['transformers'] = transformers
	ct = ColumnTransformer(**ct_parameters)
	ct.fit(X_train)

	# print(len(ct.get_feature_names()))
	X_train = ct.transform(X_train)
	X_valid = ct.transform(X_valid)

	model = CFG["model"](**CFG["parameters"])
	model.fit(X_train, y_train[:, 1])

	y_preds  = model.predict(X_valid)
	# y_preds  = np.random.randint(low = 1, high = 5, size = y_valid.shape[0])
	accuracy = accuracy_score(y_valid[:, 1], y_preds)

	oof_users.extend(y_valid[:, 0])
	oof_labels.extend(y_valid[:, 1])
	oof_predictions.extend(y_preds)
	if CFG['one_fold']: break

results = np.zeros((len(oof_users), 3))
results[:, 0] = oof_users
results[:, 1] = oof_labels
results[:, 2] = oof_predictions

if SAVE_MODEL:
	pd.DataFrame(results, columns = ["User", "Label", "Predictions"]).to_csv(f"predictions/{LEVEL}-level/{LEVEL}_results_model_oof_{ID}.csv", index = False)

OUTPUT['post_accuracy'], OUTPUT['post_precision'], OUTPUT['post_recall'], OUTPUT['post_error'] = \
	metric_evaluation(results, level = 'post', verbose = True)

OUTPUT['user_accuracy'], OUTPUT['user_precision'], OUTPUT['user_recall'], OUTPUT['user_error'] = \
	metric_evaluation(results, level = 'user', verbose = True)

GLOBAL_LOGGER.append(CFG, OUTPUT)
