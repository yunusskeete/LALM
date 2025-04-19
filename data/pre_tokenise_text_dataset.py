from typing import Optional

from datasets import load_dataset
from transformers import BertTokenizerFast

# DATASET
UNTOKENIZED_DATASET_NAME: str = "bookcorpus"
UNTOKENIZED_DATASET_SPLIT: Optional[str] = "train"
TOKENIZED_DATASET_NAME: str = "./tokenized_bookcorpus"

# MODEL
TOKENIZER_MODEL_NAME: str = "bert-base-uncased"

# TOKENISATION
BATCHED: bool = True
NUM_PROC: int = 32
MAX_LENGTH: int = 128
TRUNCATION: bool = True
PADDING: bool = False


if __name__ == "__main__":
    # Load HuggingFace dataset
    dataset = load_dataset(UNTOKENIZED_DATASET_NAME, split=UNTOKENIZED_DATASET_SPLIT)
    tokenizer = BertTokenizerFast.from_pretrained(TOKENIZER_MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=TRUNCATION,
            padding=PADDING,
            max_length=MAX_LENGTH,  # match model's max_length
        )

    # Apply pre-tokenization
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=BATCHED,
        remove_columns=["text"],  # remove raw text
        num_proc=NUM_PROC,  # multi-processing
    )

    # Set format to PyTorch
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_dataset.save_to_disk(TOKENIZED_DATASET_NAME)

    print(tokenized_dataset)
