from datasets import Dataset

# DATASET
TOKENIZED_DATASET_NAME: str = "./tokenized_bookcorpus"
TEST_SPLIT: float = 0.02
SEED: int = 42


class PreTokenizedTextDataset(Dataset):
    """
    Dataset that returns pre-tokenized tensors (input_ids and attention_mask).
    """

    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
        }


if __name__ == "__main__":
    from datasets import load_from_disk
    from torch.utils.data import DataLoader

    # 1. Load full dataset
    full_dataset = load_from_disk(TOKENIZED_DATASET_NAME)

    # 2. Split into train/val
    splits = full_dataset.train_test_split(
        test_size=TEST_SPLIT,  # 2% validation, 98% train
        seed=SEED,
    )

    # 3. Load datasets
    # Datasets
    train_dataset = PreTokenizedTextDataset(splits["train"])
    val_dataset = PreTokenizedTextDataset(splits["test"])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    print(f"{train_loader=}")
    print(f"{val_loader=}")

    # 4. Check dataloader
    for batch in train_loader:
        print(f"{batch=}")
        break
