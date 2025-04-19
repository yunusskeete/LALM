import itertools
import time
from typing import Dict, Optional

import torch
from datasets import load_dataset, load_from_disk
from torch.cuda.amp import autocast
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from x_transformers import Decoder
from x_transformers.x_transformers import ScaledSinusoidalEmbedding

from data.pre_tokenised_text_dataset import PreTokenizedTextDataset
from data.text_dataset import TextDataset
from model.lalm import LatentAutoregressiveLanguageModel

# RUNTIME
ACCELERATOR: str = "gpu"
DEVICES: int = 1
PRECISION: int = 16
WARMUP_STEPS: int = 10
STEPS: int = 50

# TRANSFORMER ENCODER/DECODER
NUM_LAYERS: int = 6  # Number of layers in the transformer
NUM_HEADS: int = 8  # Number of attention heads in the transformer
LAYER_DROPOUT: float = 0.1
# LALM
MAX_LENGTH: int = 128
LR: float = 1e-4
LR_WARMUP_FRACTION: float = 0.1
PRE_TOKENIZED_DATASET: bool = True

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


def dynamic_padding_collate(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]

    # Pad dynamically to max length in batch
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def step(
    model: LatentAutoregressiveLanguageModel,
    batch_iter: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    precision: int,
) -> None:
    try:
        batch = next(batch_iter)
    except StopIteration:
        batch_iter = iter(dataloader)
        batch = next(batch_iter)

    if isinstance(batch, dict) and "input_ids" in batch:
        token_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
    else:
        token_ids, attention_mask = model.tokenize_batch(batch)
        token_ids, attention_mask = token_ids.to(device), attention_mask.to(device)

    optimizer.zero_grad()
    with autocast(enabled=(precision == PRECISION)):
        loss, _ = model.forward(token_ids, attention_mask)
    loss.backward()
    optimizer.step()


def measure_throughput(
    model: LatentAutoregressiveLanguageModel,
    dataloader: DataLoader,
    batch_size: int,
    device: torch.device,
    precision: int = 16,
    steps: int = 50,
) -> float:
    """
    Measures iterations per second for a given DataLoader config.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    batch_iter = iter(dataloader)

    # Optional warmup
    for _ in range(WARMUP_STEPS):
        step(
            model=model,
            batch_iter=batch_iter,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            precision=precision,
        )

    start_time = time.time()

    for _ in range(steps):
        step(
            model=model,
            batch_iter=batch_iter,
            optimizer=optimizer,
            dataloader=dataloader,
            device=device,
            precision=precision,
        )

    end_time = time.time()
    elapsed = end_time - start_time
    ips = steps / elapsed

    return batch_size * ips


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

    # 2. Load dataset
    full_dataset = (
        load_dataset(UNTOKENIZED_DATASET_NAME, split=UNTOKENIZED_DATASET_SPLIT)
        if not PRE_TOKENIZED_DATASET
        else load_from_disk(TOKENIZED_DATASET_NAME)
    )

    # 3. Split into train/val
    splits = full_dataset.train_test_split(
        test_size=TEST_SPLIT,
        seed=SEED,
    )

    # Datasets
    dataset = (
        TextDataset(splits["train"])
        if not PRE_TOKENIZED_DATASET
        else PreTokenizedTextDataset(splits["train"])
    )

    # 4. Sweep Parameters
    batch_sizes = [100, 112]
    num_workers_list = [8, 16, 24]
    pin_memory_list = [True, False]
    persistent_workers_list = [True, False]
    prefetch_factors = [2, 4]

    best_config: Optional[Dict[str, int]] = None
    best_ips: float = 0.0

    log_dir = "./profiler_logs"
    writer = SummaryWriter(log_dir)

    for run_id, (
        batch_size,
        num_workers,
        pin_memory,
        persistent_workers,
        prefetch_factor,
    ) in enumerate(
        itertools.product(
            batch_sizes,
            num_workers_list,
            pin_memory_list,
            persistent_workers_list,
            prefetch_factors,
        )
    ):
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                shuffle=True,
                collate_fn=dynamic_padding_collate,
            )

            model = LatentAutoregressiveLanguageModel(
                tokenizer=tokeniser,
                embedding_layer=embedding_model_bert.embeddings,
                pos_embedding_layer=pos_embedding,
                forecaster=forecaster,
                max_length=MAX_LENGTH,
                lr=LR,
            )
            ips = measure_throughput(
                model=model,
                dataloader=dataloader,
                batch_size=batch_size,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                precision=PRECISION,
                steps=STEPS,
            )
            print(
                f"IPS: {ips:.2f} | batch_size={batch_size} workers={num_workers} pin_memory={pin_memory} persistent={persistent_workers} prefetch={prefetch_factor}"
            )

            # Log to TensorBoard
            writer.add_scalar("iterations_per_second", ips, run_id)

            config: Dict[str, int] = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": int(pin_memory),
                "persistent_workers": int(persistent_workers),
                "prefetch_factor": prefetch_factor,
            }

            # Log the hyperparameters
            writer.add_hparams(
                config,
                {"hparam/iterations_per_second": ips},
                run_name=f"run_{run_id}",
            )

            if ips > best_ips:
                best_ips = ips
                best_config = config

        except Exception as e:
            print(f"Failed config: {e}")

    writer.close()

    # 5. Print best config
    print("\n🏆 Best Configuration:")
    print(f"IPS: {best_ips:.2f}")
    print(f"{best_config=}")
