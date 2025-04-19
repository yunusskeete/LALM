from typing import Any, Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
from transformers import get_cosine_schedule_with_warmup


class LatentAutoregressiveLanguageModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        embedding_layer: nn.Module,
        pos_embedding_layer: nn.Module,
        forecaster: nn.Module,
        max_length: int = 128,
        lr: float = 1e-4,
        lr_warmup_fraction: float = 0.1,
        pre_tokenized_dataset: bool = False,
    ):
        """
        Lightning Module for latent autoregressive language modeling.

        Args:
            tokenizer: HuggingFace tokenizer.
            embedding_layer: Token embedding model (e.g., BERT embeddings).
            pos_embedding_layer: Positional embedding module.
            forecaster: Decoder model (causal Transformer).
            max_length: Maximum sequence length.
            lr: Learning rate.
            lr_warmup_fraction: Fraction of learning rate warmup steps (default: 0.1).
            pre_tokenized_dataset: Whether the dataset is pre-tokenized.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.pos_embedding_layer = pos_embedding_layer
        self.forecaster = forecaster
        self.max_length = max_length
        self.lr = lr
        self.lr_warmup_fraction = lr_warmup_fraction
        self.pre_tokenized_dataset = pre_tokenized_dataset

        self.criterion = nn.MSELoss(reduction="none")

    def _tokenize_batch(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a batch of texts and return the token IDs and attention mask.

        Args:
            texts: A list of strings to tokenize.

        Returns:
            A tuple of two tensors: the first containing the input token IDs and the
            second containing the attention mask.
        """
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        return encoding["input_ids"], encoding["attention_mask"]

    def _get_tokens_from_batch(
        self, batch: Union[Dict[str, torch.Tensor], List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a batch of raw texts or pre-tokenized tensors into a pair of tensors
        representing the input token IDs and attention mask.

        Args:
            batch: A list of strings or a dictionary of tensors containing pre-tokenized
                input IDs and attention masks.

        Returns:
            A tuple of two tensors: the first containing the input token IDs and the
            second containing the attention mask.
        """
        if self.pre_tokenized_dataset or isinstance(batch, dict):
            token_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

        elif isinstance(batch, list):
            token_ids, attention_mask = self._tokenize_batch(batch)
            token_ids, attention_mask = token_ids.to(self.device), attention_mask.to(
                self.device
            )

        else:
            raise ValueError("Unsupported batch type.")

        return token_ids, attention_mask

    def _embed_with_positional(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input token IDs with positional embeddings.

        Args:
            token_ids: A tensor of shape `(batch_size, seq_len)` containing input token IDs.

        Returns:
            A tensor of shape `(batch_size, seq_len, embed_dim)` containing the embedded token IDs with positional embeddings.
        """
        embeddings = self.embedding_layer(token_ids)
        embeddings = embeddings + self.pos_embedding_layer(embeddings)

        return embeddings

    def _shift(
        self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shifts the predictions, targets, and mask for next prediction.
        The predictions and targets are shifted by one position to the right, effectively
        removing the first prediction and target, and the last mask value.

        Args:
            preds: A tensor of shape `(batch_size, seq_len, vocab_size)` containing the
                predictions.
            targets: A tensor of shape `(batch_size, seq_len)` containing the targets.
            mask: A tensor of shape `(batch_size, seq_len)` containing the mask.

        Returns:
            A tuple of three tensors: the shifted predictions, targets, and mask.
        """

        return preds[:, :-1], targets[:, 1:], mask[:, 1:]

    def _masked_mse_loss(
        self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the masked mean squared error loss between the predictions and targets.

        Args:
            preds: A tensor of shape `(batch_size, seq_len, vocab_size)` containing the predictions.
            targets: A tensor of shape `(batch_size, seq_len)` containing the targets.
            mask: A tensor of shape `(batch_size, seq_len)` containing the mask.

        Returns:
            A tensor containing the masked mean squared error loss.
        """
        per_token_loss = self.criterion(preds, targets)
        per_token_loss = per_token_loss.mean(dim=-1)
        masked_loss = per_token_loss * mask

        return masked_loss.sum() / mask.sum()

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Carries out a single forward pass.

        Args:
            token_ids: A tensor of shape `(batch_size, seq_len)` containing input token IDs.
            attention_mask: A tensor of shape `(batch_size, seq_len)` containing the attention mask.

        Returns:
            A tuple of two tensors: the loss and the predictions.
        """

        token_embeddings = self._embed_with_positional(token_ids)
        preds = self.forecaster(token_embeddings)
        preds, targets, mask = self._shift(preds, token_embeddings, attention_mask)
        loss = self._masked_mse_loss(preds, targets, mask)

        return loss, preds

    def training_step(
        self,
        batch: Union[Dict[str, torch.Tensor], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Carries out a single training step.

        Args:
            batch: A list of strings or a dictionary of tensors containing pre-tokenized
                input IDs and attention masks.
            batch_idx: The index of the batch in the dataset.

        Returns:
            The loss of this training step.
        """
        # Extract token IDs and attention masks
        token_ids, attention_mask = self._get_tokens_from_batch(batch)

        # Forward pass
        loss, _ = self.forward(token_ids, attention_mask)

        # Logging
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=len(batch),
        )
        self.log(
            "lr",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            on_epoch=False,
            on_step=True,
        )

        return loss

    def validation_step(
        self,
        batch: List[str],
        # batch: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Carries out a single validation step.

        Args:
            batch: A list of strings or a dictionary of tensors containing pre-tokenized
                input IDs and attention masks.
            batch_idx: The index of the batch in the validation set.

        Returns:
            The validation loss of the current batch.
        """
        # Extract token IDs and attention masks
        token_ids, attention_mask = self._get_tokens_from_batch(batch)

        # Forward pass
        loss, _ = self.forward(token_ids, attention_mask)

        # Logging
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(batch),
        )

        return loss

    def test_step(
        self,
        batch: Union[Dict[str, torch.Tensor], List[str]],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Carries out a single test step.

        Args:
            batch: A list of strings or a dictionary of tensors containing pre-tokenized
                input IDs and attention masks.
            batch_idx: The index of the batch in the test set.

        Returns:
            The test loss of the current batch.
        """
        # Extract token IDs and attention masks
        token_ids, attention_mask = self._get_tokens_from_batch(batch)

        # Forward pass
        loss, _ = self.forward(token_ids, attention_mask)

        # Logging
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(batch),
        )

        return loss

    def predict_step(
        self,
        batch: Union[Dict[str, torch.Tensor], List[str]],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Runs inference on a batch and returns the predicted latent representations.

        Args:
            batch: A list of strings or a dictionary of tensors containing pre-tokenized
                input IDs and attention masks.
            batch_idx: The index of the batch in the prediction set.
            dataloader_idx: The index of the dataloader if multiple are used.

        Returns:
            The predicted latent representations for the batch.
        """
        # Extract token IDs
        token_ids, _ = self._get_tokens_from_batch(batch)

        # Forward pass
        token_embeddings = self._embed_with_positional(token_ids)
        preds = self.forecaster(token_embeddings)

        return preds

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """
        Configures the optimizer and learning rate scheduler for training.

        The optimizer is set to Adam with the learning rate specified in
        `self.lr`. The learning rate scheduler is set to a cosine schedule with
        warmup, with the warmup period set as a fraction of the total training steps.

        The total number of training steps is computed dynamically based on the
        size of the training set and the number of epochs specified in the
        Trainer.

        Returns:
            A tuple of the optimizer and the scheduler (each wrapped in a list).
        """
        optimizer = torch.optim.Adam(self.forecaster.parameters(), lr=self.lr)

        # Dynamically compute total steps
        train_loader = (
            self.trainer.datamodule.train_dataloader()
            if self.trainer.datamodule
            else self.trainer.train_dataloader
        )
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.trainer.max_epochs

        print(f"Total training steps: {total_steps}")

        # Warmup: fraction of total steps (default: 10% of total steps)
        warmup_steps = int(self.lr_warmup_fraction * total_steps)

        print(f"Number of warmup steps: {warmup_steps}")

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scheduler = {
            "scheduler": scheduler,
            "interval": "step",  # Update every training step
            "frequency": 1,
        }

        return [optimizer], [scheduler]
