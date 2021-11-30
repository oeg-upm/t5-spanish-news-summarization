import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap

from transformers import (AdamW, MT5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer)
from tqdm.auto import tqdm

class NewsSummaryDataset(Dataset):

  def __init__(self, data: pd.DataFrame, tokenizer: T5Tokenizer, text_max_token_len: int =512, summary_max_token_len: int=128):
    self.tokenizer=tokenizer
    self.data=data
    self.text_max_token_len= text_max_token_len
    self.summary_max_token_len= summary_max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row =self.data.iloc[index]

    text=data_row["text"]

    text_encoding = self.tokenizer(text, max_length= self.text_max_token_len, padding = "max_length", truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")
    summary_encoding = self.tokenizer(data_row["summary"], max_length= self.summary_max_token_len, padding = "max_length", truncation=True, return_attention_mask=True, add_special_tokens=True, return_tensors="pt")

    labels= summary_encoding["input_ids"]
    labels[labels==0] = -100

    return dict( text= text, summary = data_row["summary"], text_input_ids=text_encoding["input_ids"].flatten(), text_attention_mask=text_encoding["attention_mask"].flatten(), labels =labels.flatten(), labels_attention_mask=summary_encoding["attention_mask"].flatten())

class NewsSummaryDataModule(pl.LightningDataModule):

  def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, val_df: pd.DataFrame, tokenizer: T5Tokenizer, batch_size: int=8, text_max_token_len: int=512, summary_max_token_len: int=128):
     super().__init__()
     self.train_df=train_df
     self.test_df=test_df
     self.val_df=val_df
     self.tokenizer=tokenizer
     self.batch_size=batch_size
     self.text_max_token_len=text_max_token_len
     self.summary_max_token_len=summary_max_token_len

  def setup(self, stage=None):
    self.train_dataset=NewsSummaryDataset(self.train_df, self.tokenizer, self.text_max_token_len, self.summary_max_token_len)
    self.test_dataset=NewsSummaryDataset(self.test_df, self.tokenizer, self.text_max_token_len, self.summary_max_token_len)
    self.val_dataset=NewsSummaryDataset(self.val_df, self.tokenizer, self.text_max_token_len, self.summary_max_token_len)
  
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

  def val_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

  def test_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)


class NewsSummaryModel(pl.LightningModule):

  def __init__(self):
    super().__init__()
    MODEL_NAME="./espt5-large"

    self.model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)

  def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
    output=self.model(input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
    return output.loss, output.logits

  def training_step(self, batch, batch_idx):
    input_ids=batch["text_input_ids"]
    attention_mask=batch["text_attention_mask"]
    labels=batch["labels"]
    labels_attention_mask=batch["labels_attention_mask"]
    loss, outputs= self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=labels_attention_mask)
    self.log("train_loss", loss, prog_bar=True, logger= True)
    return loss

  def validation_step(self, batch, batch_idx):
    input_ids=batch["text_input_ids"]
    attention_mask=batch["text_attention_mask"]
    labels=batch["labels"]
    labels_attention_mask=batch["labels_attention_mask"]
    loss, outputs= self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=labels_attention_mask)
    self.log("val_loss", loss, prog_bar=True, logger= True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids=batch["text_input_ids"]
    attention_mask=batch["text_attention_mask"]
    labels=batch["labels"]
    labels_attention_mask=batch["labels_attention_mask"]
    loss, outputs= self(input_ids=input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=labels_attention_mask)
    self.log("test_loss", loss, prog_bar=True, logger= True)
    return loss

  def configure_optimizers(self):
    return AdamW(self.parameters(), lr=0.0001)

def main():
  df = pd.read_json('./data/XLsum/spanish_train.jsonl', lines=True)
  df=df[[ "text","summary"]]
  df.dropna()
  df1 = pd.read_json('./data/XLsum/spanish_test.jsonl', lines=True)
  df1=df[[ "text","summary"]]
  df1.dropna()
  df2 = pd.read_json('./data/XLsum/spanish_val.jsonl', lines=True)
  df2=df[[ "text","summary"]]
  df2.dropna()
  df=df
  df1=df1
  df2=df2
  train_df=df
  test_df=df1
  val_df=df2
  MODEL_NAME="./espt5-large"
  tokenizer=T5Tokenizer.from_pretrained(MODEL_NAME)
  N_EPOCHS=2
  BATCH_SIZE=2

  data_module = NewsSummaryDataModule(train_df,test_df,val_df,tokenizer,batch_size=BATCH_SIZE)
  model = NewsSummaryModel()
  
  checkpoint_callback= ModelCheckpoint(dirpath= "checkpoints", filename="", save_top_k=2, verbose=True, monitor="val_loss", mode="min")
  logger = TensorBoardLogger("lightning_logs", name="news-summary")  
  trainer = pl.Trainer( logger=logger, callbacks=checkpoint_callback, max_epochs= N_EPOCHS, gpus=0, progress_bar_refresh_rate=30)
  trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
