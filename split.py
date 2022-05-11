from utils import *

dataset = suicide_risk_assessment(source = None)
dataset['label_map'] = dataset['label'].map({"Supportive" : 1, "Indicator" : 2, "Ideation" : 3, "Behavior" : 4, "Attempt" : 5})

# display(dataset)

# plt.figure(figsize = (24, 10))
# plt.bar(*np.unique(dataset['user'], return_counts = True))
# plt.title("Post Grouped by Users")
# plt.show()

# plt.figure(figsize = (24, 10))
# plt.bar(*np.unique(dataset['label'], return_counts = True))
# plt.title("Label Distribution")
# plt.show()

values, counts = np.unique(dataset['user'], return_counts = True)
ones = values[counts == 1]

dataset['only_one'] = 0
dataset.loc[dataset['user'].isin(ones), 'only_one'] = 1

dataset["1_fold"] = -1
dataset["2_fold"] = -1

single_posts   = dataset[dataset['only_one'] == 1].reset_index(drop = True)
multiple_posts = dataset[dataset['only_one'] == 0].reset_index(drop = True)

# print(len(np.unique(dataset[dataset['only_one'] == 1]['user'], return_counts = False)))
# plt.figure(figsize = (24, 10))
# plt.bar(*np.unique(dataset[dataset['only_one'] == 1]['label'], return_counts = True))
# plt.title("Label Distribution")
# plt.show()

# print(*np.unique(multiple_posts['label'], return_counts = True))
# plt.figure(figsize = (12, 5))
# plt.bar(*np.unique(multiple_posts['label'], return_counts = True))
# plt.title("Label Distribution")
# plt.show()


threshold = multiple_posts.shape[0] * 0.2

found = True
counters = [0, 0, 0, 0, 0]

while found:
	found = False

	# We want all folds to have all classes
	# For each fold we take one available user for every label
	# We sort the groups for speed and better statification
	# We stop when we don't have users
	for fold in range(5):
		if counters[fold] > threshold: continue

		for label in np.unique(multiple_posts['label']):
			label_skf = multiple_posts[multiple_posts["label"] == label]

			for idx, (group_idx, group) in enumerate(sorted(label_skf.groupby(["user"]), 
										key = lambda x: len(x[1]), 
										reverse = True)):

				if group['1_fold'].values[0] == -1:
					multiple_posts.loc[group.index, '1_fold'] = fold
					counters[fold] += len(group)
					found = True
					break


# for fold in range(5):
# 	print(f"Fold: {fold}")
# 	plt.figure(figsize = (12, 5))
# 	print(*np.unique(multiple_posts[multiple_posts['1_fold'] == fold]['label'], return_counts = True))
# 	plt.bar(*np.unique(multiple_posts[multiple_posts['1_fold'] == fold]['label'], return_counts = True))
# 	plt.title("Label Distribution")
# 	plt.show()


print(*np.unique(multiple_posts['1_fold'], return_counts = True))
print(100 * "=")

indexes, randoms = [], [] 
folds_idxs = [[], [], [], [], []]
for idx, (group_idx, group) in enumerate(sorted(multiple_posts.groupby(["user"]), 
										key = lambda x: len(x[1]), 
										reverse = True)):
	
	if len(group) < 5: 
		indexes.extend(group.index)
		randoms.extend(random.sample(range(5), len(group)))
		continue

	kf = KFold(n_splits = 5)
	kf.get_n_splits(group) 

	for fold, (train_idx, valid_idx) in enumerate(kf.split(group)):
		folds_idxs[fold].extend(group.iloc[valid_idx].index)

for fold, fold_idx in enumerate(folds_idxs):
	multiple_posts.loc[fold_idx, '2_fold'] = fold

multiple_posts.loc[indexes, '2_fold'] = randoms

# for fold in range(5):
# 	print(f"Fold: {fold}")
# 	plt.figure(figsize = (12, 5))
# 	print(*np.unique(multiple_posts[multiple_posts['2_fold'] == fold]['label'], return_counts = True))
# 	plt.bar(*np.unique(multiple_posts[multiple_posts['2_fold'] == fold]['label'], return_counts = True))
# 	plt.title("Label Distribution")
# 	plt.show()

display(*np.unique(multiple_posts['2_fold'], return_counts = True))

dataset = pd.concat([single_posts, multiple_posts], axis = 0)
dataset = dataset.sample(frac = 1, random_state = SEED).reset_index(drop = True)

display(*np.unique(dataset['1_fold'], return_counts = True))
display(*np.unique(dataset['2_fold'], return_counts = True))

dataset['1_fold'] = dataset['1_fold'].astype(int)
dataset['2_fold'] = dataset['2_fold'].astype(int)

dataset = dataset.drop(["label_map"], axis = 1, inplace = False)

# display(dataset.iloc[2100].text)
# dataset.to_csv("suicide_squad_test.csv", index = False)