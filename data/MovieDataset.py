import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        """
        Initializes the MovieDataset class.

        Args:
            texts (list): List of texts.
            labels (list): List of corresponding labels.
            tokenizer: Tokenizer object to tokenize the texts.
            max_length (int): Maximum length of the tokenized sequences.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Retrieves a single item (text, label) from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing the tokenized input_ids, attention_mask, and label.
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text using the tokenizer
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)

        # Flatten and convert input_ids and attention_mask to tensors, and convert label to tensor
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(int(label))}
