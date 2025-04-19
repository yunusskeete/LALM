from typing import Dict, List, Optional

import torch
from datasets import load_dataset, load_from_disk
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
from x_transformers import Decoder
from x_transformers.x_transformers import ScaledSinusoidalEmbedding

from data.pre_tokenised_text_dataset import PreTokenizedTextDataset
from data.text_dataset import TextDataset
from model.lalm import LatentAutoregressiveLanguageModel

# TRANSFORMER ENCODER/DECODER
NUM_LAYERS: int = 6  # Number of layers in the transformer
NUM_HEADS: int = 8  # Number of attention heads in the transformer
LAYER_DROPOUT: float = 0.1
# LALM
MAX_LENGTH: int = 128
LR: float = 1e-4
LR_WARMUP_FRACTION: float = 0.1
PRE_TOKENIZED_DATASET: bool = True

# RUNTIME
ACCELERATOR: str = "gpu"
DEVICES: int = 1
PRECISION: int = 16
MODEL_SUMMARY_MAX_DEPTH: int = 3
VAL_CHECK_INTERVAL: float = 0.25
MAX_EPOCHS: int = 5
BATCH_SIZE: int = 100
NUM_WORKERS: int = 16
PIN_MEMORY: bool = True
PERSISTENT_WORKERS: bool = True
PREFETCH_FACTOR: int = 2

# DATASET
UNTOKENIZED_DATASET_NAME: str = "bookcorpus"
UNTOKENIZED_DATASET_SPLIT: Optional[str] = "train"
TOKENIZED_DATASET_NAME: str = "./tokenized_bookcorpus"
SHUFFLE_TRAIN_DATASET: bool = True
TEST_SPLIT: float = 0.02
SEED: int = 42

# MODEL
LALM_MODEL_NAME: str = "lalm"
EMBEDDING_MODEL_NAME: str = "bert-base-uncased"
TOKENIZER_MODEL_NAME: str = "bert-base-uncased"

PAD_TOKEN_ID: int = 0


def dynamic_padding_collate(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collates a batch of data by dynamically padding the sequences to match the
    maximum length of the sequences in the batch.

    Args:
        batch: A list of dictionaries, where each dictionary contains 'input_ids'
            and 'attention_mask' as keys with their corresponding tensor values.

    Returns:
        A dictionary containing padded 'input_ids' and 'attention_mask' tensors.
        The sequences are padded to the length of the longest sequence in the batch.
    """
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Pad dynamically to max length in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=PAD_TOKEN_ID
    )

    return {"input_ids": input_ids, "attention_mask": attention_mask}


if __name__ == "__main__":
    # 1. Define model
    # Load pretrained BERT tokeniser
    tokeniser = BertTokenizer.from_pretrained(TOKENIZER_MODEL_NAME)
    embedding_model_bert: BertModel = BertModel.from_pretrained(EMBEDDING_MODEL_NAME)

    PAD_TOKEN_ID: int = tokeniser.pad_token_id

    # TOKENISATION - BERT vocab size (30522)
    VOCAB_SIZE: int = tokeniser.vocab_size
    # TOKEN EMBEDDING - BERT hidden dimension (768)
    EMBED_DIM: int = embedding_model_bert.config.hidden_size

    pos_embedding = ScaledSinusoidalEmbedding(EMBED_DIM)

    forecaster = Decoder(
        dim=EMBED_DIM,
        depth=NUM_LAYERS,
        heads=NUM_HEADS,
        layer_dropout=LAYER_DROPOUT,
        # Causal attn
    )

    model = LatentAutoregressiveLanguageModel(
        tokenizer=tokeniser,
        embedding_layer=embedding_model_bert.embeddings,
        pos_embedding_layer=pos_embedding,
        forecaster=forecaster,
        max_length=MAX_LENGTH,
        lr=LR,
        lr_warmup_fraction=LR_WARMUP_FRACTION,
        pre_tokenized_dataset=PRE_TOKENIZED_DATASET,
    )

    # 2. Load dataset
    full_dataset = (
        load_dataset(UNTOKENIZED_DATASET_NAME, split=UNTOKENIZED_DATASET_SPLIT)
        if not PRE_TOKENIZED_DATASET
        else load_from_disk(TOKENIZED_DATASET_NAME)
    )

    # 3. Split into train/val
    splits = full_dataset.train_test_split(
        test_size=TEST_SPLIT,  # 2% validation, 98% train
        seed=SEED,
    )

    # Datasets
    train_dataset = (
        TextDataset(splits["train"])
        if not PRE_TOKENIZED_DATASET
        else PreTokenizedTextDataset(splits["train"])
    )
    val_dataset = (
        TextDataset(splits["test"])
        if not PRE_TOKENIZED_DATASET
        else PreTokenizedTextDataset(splits["test"])
    )

    # 4. Train
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        shuffle=SHUFFLE_TRAIN_DATASET,
        collate_fn=dynamic_padding_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        collate_fn=dynamic_padding_collate,
    )

    trainer = Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator=ACCELERATOR,
        devices=DEVICES,
        precision=PRECISION,
        callbacks=[
            ModelSummary(max_depth=MODEL_SUMMARY_MAX_DEPTH),
        ],
        val_check_interval=VAL_CHECK_INTERVAL,
    )

    trainer.fit(model, train_loader, val_loader)

    # 5. Save
    trainer.save_checkpoint(f"{LALM_MODEL_NAME}.ckpt")

    # 6. Test
    trainer.test(model, dataloaders=val_loader)
