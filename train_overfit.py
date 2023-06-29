import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, LongT5ForConditionalGeneration

from lightning import Trainer, Callback
# from lightning.callbacks import EarlyStopping, LearningRateMonitor

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import lightning as pl


# torch.set_float32_matmul_precision('medium')

class MyCallback(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.val_accuracy = 0
    
    def on_validation_epoch_end(self, trainer, pl_module):
      print("Accuracy:", pl_module.val_accuracy)
        

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--overfit_test",
        action="store_true",
        help="Flag to start a normal training or an overfitting test.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        required=True,
        help="Path to tsv file.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    return args



class MyDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return self.data["input_ids"].shape[0]

  def __getitem__(self, idx):
    inputs = {
        "attention_mask": self.data["attention_mask"][idx],
        "input_ids": self.data["input_ids"][idx],
        "labels": self.data["labels"][idx]
    }
    return inputs


def preprocess_examples(input_texts, labels, tokenizer):

  model_inputs = tokenizer(input_texts, padding="longest", return_tensors="pt")
  labels = tokenizer(labels, padding="longest").input_ids

  # important: we need to replace the index of the padding tokens by -100
  # such that they are not taken into account by the CrossEntropyLoss
  labels_with_ignore_index = []
  for labels_example in labels:
    labels_example = [label if label != 0 else -100 for label in labels_example]
    labels_with_ignore_index.append(labels_example)

  inputs = {
      "attention_mask": model_inputs.attention_mask,
      "input_ids": model_inputs.input_ids,
      "labels": torch.tensor(labels_with_ignore_index)
  }

  return inputs


class ListwiseT5(pl.LightningModule):
    def __init__(self, pretrained_model_name_or_path, tokenizer):
        super().__init__()
        self.model = LongT5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path) # "google/long-t5-tglobal-base"
        self.tokenizer = tokenizer
        self.val_accuracy = 0
        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        outputs = self.model.generate(batch["input_ids"], max_length=200, do_sample=False)
        pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        target = self.tokenizer.decode(torch.where(batch["labels"][0] != -100, batch["labels"][0], 0), skip_special_tokens=True)
        self.val_accuracy += int(pred == target)
        print(pred, "/", target, "/", loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=3e-4)
        scheduler = {
            "scheduler": CosineAnnealingWarmRestarts(optimizer, 1),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    df = pd.read_csv(args.data_path, sep="\t")
    end_train_data = int(len(df) * 0.8)
    train_data = preprocess_examples(list(df["prompt"])[:end_train_data], list(df["target"])[:end_train_data], tokenizer)
    valid_data = preprocess_examples(list(df["prompt"])[end_train_data:], list(df["target"])[end_train_data:], tokenizer)

    train_dataset = MyDataset(train_data)
    valid_dataset = MyDataset(valid_data)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    model = ListwiseT5(args.pretrained_model_name_or_path, tokenizer)
    callback = MyCallback()
    trainer = Trainer(accelerator="auto",
                      enable_checkpointing=False,
                      # log_every_n_steps=1,
                      # default_root_dir="checkpoints",
                      overfit_batches=5 if args.overfit_test else 0,
                      accumulate_grad_batches=5,
                      max_epochs=-1,
                      callbacks=[callback])

    trainer.fit(model, train_dataloader, train_dataloader)

if __name__ == "__main__":
    args = parse_args()
    main(args)