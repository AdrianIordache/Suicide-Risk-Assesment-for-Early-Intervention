from utils import *

from utils import *
from preprocessing import *

GLOBAL_LOGGER = GlobalLogger(
    path_to_global_logger = f'logs/bayesian_logger_oof_users.csv',
    save_to_log = True
)

MODELS = [
    (LGBMClassifier, {"n_estimators" : 50 , "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 100, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 200, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 250, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 300, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 350, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 450, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 500, "random_state" : SEED}),
    (LGBMClassifier, {"n_estimators" : 100, "random_state" : SEED, "num_leaves" : 64}),
    (LGBMClassifier, {"n_estimators" : 200, "random_state" : SEED, "num_leaves" : 64}),
    (LGBMClassifier, {"n_estimators" : 300, "random_state" : SEED, "num_leaves" : 64}),
    (LGBMClassifier, {"n_estimators" : 300, "random_state" : SEED, "num_leaves" : 32,  "colsample_bytree" : 0.8}),
    (LGBMClassifier, {"n_estimators" : 300, "random_state" : SEED, "num_leaves" : 64,  "colsample_bytree" : 0.8}),
    (LGBMClassifier, {"n_estimators" : 300, "random_state" : SEED, "num_leaves" : 128, "colsample_bytree" : 0.8}),

    (SVC, {"C" : 0.1, "random_state": SEED}),
    (SVC, {"C" : 0.5, "random_state": SEED}),
    (SVC, {"C" : 0.7, "random_state": SEED}),
    (SVC, {"C" : 1.0, "random_state": SEED}),
    (SVC, {"C" : 2.0, "random_state": SEED}),
    (SVC, {"C" : 3.0, "random_state": SEED}),
    (SVC, {"C" : 5.0, "random_state": SEED}),
    (SVC, {"C" : 7.0, "random_state": SEED}),
    (SVC, {"C" : 10.0, "random_state": SEED}),
    (SVC, {"C" : 15.0, "random_state": SEED}),

    (AdaBoostClassifier, {"n_estimators" : 25 , "random_state" : SEED}),
    (AdaBoostClassifier, {"n_estimators" : 50 , "random_state" : SEED}),
    (AdaBoostClassifier, {"n_estimators" : 75 , "random_state" : SEED}),
    (AdaBoostClassifier, {"n_estimators" : 100 , "random_state" : SEED}),
    (AdaBoostClassifier, {"n_estimators" : 200 , "random_state" : SEED}),

    (RandomForestClassifier, {"n_estimators" : 50 , "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 100, "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 200, "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 250, "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 300, "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 350, "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 450, "random_state" : SEED}),
    (RandomForestClassifier, {"n_estimators" : 500, "random_state" : SEED}),

    (LogisticRegression, {"C" : 0.1, "random_state": SEED}),
    (LogisticRegression, {"C" : 0.5, "random_state": SEED}),
    (LogisticRegression, {"C" : 0.7, "random_state": SEED}),
    (LogisticRegression, {"C" : 1.0, "random_state": SEED}),
    (LogisticRegression, {"C" : 2.0, "random_state": SEED}),
    (LogisticRegression, {"C" : 3.0, "random_state": SEED}),
    (LogisticRegression, {"C" : 5.0, "random_state": SEED}),
    (LogisticRegression, {"C" : 7.0, "random_state": SEED}),
    (LogisticRegression, {"C" : 10.0, "random_state": SEED}),
    (LogisticRegression, {"C" : 15.0, "random_state": SEED}),
]

COLUMNS = [
    ["text"],
    ["social_prepocessed_text"],
    ['text', 'happy', 'angry', 'surprise', 'sad', 'fear'],
    ['social_prepocessed_text', 'happy', 'angry', 'surprise', 'sad', 'fear'],
]


TRANSFORMERS = [
    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 1), "token_pattern": r'(?u)\b\w\w+\b|!|,|.|\?|\"|\''},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 2), "token_pattern": r'(?u)\b\w\w+\b|!|,|.|\?|\"|\''},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 3), "token_pattern": r'(?u)\b\w\w+\b|!|,|.|\?|\"|\''},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 5), "token_pattern": r'(?u)\b\w\w+\b|!|,|.|\?|\"|\''},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 7), "token_pattern": r'(?u)\b\w\w+\b|!|,|.|\?|\"|\''},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 1)},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 2)},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 3)},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 5)},  "columns": 'text'}],

    [{"name" : 'word_tfidf', "algorithm": TfidfVectorizer, "parameters": {"analyzer": "word", "ngram_range": (1, 7)},  "columns": 'text'}],
]

logger = Logger("logs/logger_OOF_users.log", distributed = True)

@ignore_warnings(category = ConvergenceWarning)
def train(model_idx, columns_idx, transformers_idx):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

    model        = MODELS[math.floor(model_idx)]
    columns      = COLUMNS[math.floor(columns_idx)] 
    transformers = TRANSFORMERS[math.floor(transformers_idx)]

    transformers[0]["columns"] = copy.deepcopy(columns[0])
    
    # print(model)
    # print(columns)
    # print(transformers)

    CFG = {
        "id": GLOBAL_LOGGER.get_version_id(),
        "model": model[0],
        "parameters": model[1],
        "columns": columns,
        "transformers": transformers,
        "column_transforms": {
            'remainder': StandardScaler(),
            'n_jobs': -1
        },
        "valid_strategy": 1,
        "one_fold": False,
        "observation": "BayesianOptimizationOOF"
    }


    dataset = pd.read_csv("data/suicide_users_preprocessed.csv")

    dataset['label']    = dataset['label'].map({"Supportive" : 1, "Indicator" : 2, "Ideation" : 3, "Behavior" : 4, "Attempt" : 5})
    dataset['w_counts'] = dataset['prepocessed_text'].apply(lambda x: len(word_tokenize(str(x))))

    logger.print(f"Config File: {CFG}")
    oof_users, oof_labels, oof_predictions = [], [], [] 
    for fold in range(5):
        logger.print("=" * 75 + f" FOLD {fold} " + "=" * 75)
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

        X_train = ct.transform(X_train)
        X_valid = ct.transform(X_valid)

        model = CFG["model"](**CFG["parameters"])
        model.fit(X_train, y_train[:, 1])

        y_preds  = model.predict(X_valid)
        accuracy = accuracy_score(y_valid[:, 1], y_preds)

        oof_users.extend(y_valid[:, 0])
        oof_labels.extend(y_valid[:, 1])
        oof_predictions.extend(y_preds)
        if CFG['one_fold']: break

    results = np.zeros((len(oof_users), 3))
    results[:, 0] = oof_users
    results[:, 1] = oof_labels
    results[:, 2] = oof_predictions

    OUTPUT['post_accuracy'], OUTPUT['post_precision'], OUTPUT['post_recall'], OUTPUT['post_error'] = \
        metric_evaluation(results, level = 'post', verbose = False)

    OUTPUT['user_accuracy'], OUTPUT['user_precision'], OUTPUT['user_recall'], OUTPUT['user_error'] = \
        metric_evaluation(results, level = 'user', verbose = False)

    GLOBAL_LOGGER.append(CFG, OUTPUT)

    logger.print(f"[Level: post] Accuracy: {OUTPUT['post_accuracy']}, Precision: {OUTPUT['post_precision']}, Recall: {OUTPUT['post_recall']}, Ordinal Error: {OUTPUT['post_error']}")
    logger.print(f"[Level: user] Accuracy: {OUTPUT['user_accuracy']}, Precision: {OUTPUT['user_precision']}, Recall: {OUTPUT['user_recall']}, Ordinal Error: {OUTPUT['user_error']}")

    return OUTPUT['user_accuracy']

def Optimize(bounds, init_points = 32, iterations = 64):
    tic = time.time()
    optimizer = BayesianOptimization(train, bounds, random_state = SEED)
    optimizer.maximize(init_points = init_points, n_iter = iterations, acq = 'ucb', xi = 0.0, alpha = 1e-6)

    results = open("Optimized_Parameters_OOF_Users.pickle","wb")
    pickle.dump(optimizer.max, results)
    results.close()

    toc = time.time()
    print("Time to optimize {}'s'".format(toc - tic))


if __name__ == "__main__":

    bounds = {
        "model_idx": (0, 46.99),
        "columns_idx": (0, 3.99),
        "transformers_idx": (0, 9.99),
    }

    Optimize(bounds, init_points = 128, iterations = 512)