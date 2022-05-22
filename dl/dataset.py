import torch

labels_map = {
    "Supportive" : 0,
    "Indicator" : 1,
    "Ideation" : 2,
    "Behavior" : 3,
    "Attempt" : 4
}
labels_map_inverse = {
    0: "Supportive",
    1: "Indicator",
    2: "Ideation",
    3: "Behavior",
    4: "Attempt"
}

class SuicideDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
#         self.labels = labels
        self.labels = [labels_map[label] for label in labels]

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)