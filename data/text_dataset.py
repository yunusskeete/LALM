from typing import Optional

from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

# DATASET
UNTOKENIZED_DATASET_NAME: str = "bookcorpus"
UNTOKENIZED_DATASET_SPLIT: Optional[str] = "train"
TEST_SPLIT: float = 0.02
SEED: int = 42


class TextDataset(TorchDataset):
    """
    Wraps a HuggingFace Dataset to return text strings.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> str:
        try:
            text = self.dataset[idx]["text"]
            assert isinstance(text, str), "Sample is not a string."
            return text
        except Exception as e:
            print(f"[WARNING] Bad sample at index {idx}: {e}")
            # Return a dummy sample that will be filtered later
            return None


if __name__ == "__main__":
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # 1. Load full dataset
    full_dataset = load_dataset(
        UNTOKENIZED_DATASET_NAME, split=UNTOKENIZED_DATASET_SPLIT
    )

    # 2. Split into train/val
    splits = full_dataset.train_test_split(
        test_size=TEST_SPLIT,  # 2% validation, 98% train
        seed=SEED,
    )

    # 3. Load datasets
    # Datasets
    train_dataset = TextDataset(splits["train"])
    val_dataset = TextDataset(splits["test"])

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)

    print(f"{train_loader=}")
    print(f"{val_loader=}")

    # 4. Check dataloader
    for batch in train_loader:
        print(f"{batch=}")
        break
