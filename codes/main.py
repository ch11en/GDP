import argparse
import os

import json
import numpy as np
import parse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from data_utils import GDPDATASET, read_line_examples_from_file


from utils import LoggingCallback

from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

from model_gdp import LightningModule

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default='gdp', type=str)
    parser.add_argument("--dataset", default='Laptop', type=str, help="choice: [Laptop, Restaurant, Rest15, Rest16]")
    parser.add_argument("--model_name_or_path", default='t5_base', type=str, help="replace you path of base model")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_infer", default=True, help="Whether to run inference with trained checkpoints")

    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--stat_full_train_ep", type=int, default=-1)
    parser.add_argument("--early_stopping", type=int, default=0)
    parser.add_argument("--cont_loss", type=float, default=0.0)
    parser.add_argument("--cont_temp", type=float, default=0.25)
    parser.add_argument('--truncate', action='store_true')
    parser.add_argument("--grid_loss_lambda", type=float, default=0.1)
    parser.add_argument("--scl_loss_lambda", type=float, default=0.2)
    parser.add_argument("--category_loss_lambda", type=float, default=0.0)

    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_seq_length", default=128, type=int)

    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--pad_value", default=32099, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--df_num", default=5, type=int)
    parser.add_argument("--m", default=-0.6, type=float)
    parser.add_argument("--gama", default=10, type=int)
    parser.add_argument("--p", default=0.4, type=float)

    parser.add_argument("--num_train_epochs", default=20, type=int)
    parser.add_argument("--gpus", default=[1])
    parser.add_argument("--do_template_gen", default=False, type=bool)
    parser.add_argument("--output_dir", default='../outputs', type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)

    args = parser.parse_args()

    output_params = "_".join(['dataset', args.dataset,
                              'seed', str(args.seed),
                              'lr', str(args.learning_rate),])

    output_dir = Path(args.output_dir) / output_params
    print(f'output_dir: {output_dir}')

    if not os.path.exists(output_dir):
        output_dir.mkdir(parents=True, exist_ok=True)

    args.output_dir = output_dir

    seed_everything(args.seed, workers=True)

    return args


if __name__ == "__main__":

    args = init_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(['[SSEP]'])

    dataset = GDPDATASET(tokenizer=tokenizer, data_dir=args.dataset, max_len=args.max_seq_length,
                         data_type='train', task=args.task, truncate=args.truncate)

    model = LightningModule(args, args.model_name_or_path, tokenizer)

    train_params = dict(default_root_dir=args.output_dir,
                        accumulate_grad_batches=args.gradient_accumulation_steps,
                        accelerator='gpu',
                        devices=args.gpus,
                        gradient_clip_val=1.0,
                        max_epochs=args.num_train_epochs,
                        auto_lr_find=False,
                        deterministic=True,
                        callbacks=[LoggingCallback()],
                        strategy="dp")
    trainer = pl.Trainer(**train_params)
    if args.do_train:
        trainer.fit(model)
        trainer.save_checkpoint(os.path.join(args.output_dir, "final_model.ckpt"))

    if args.do_infer:
        trainer.test(model, dataloaders=model.test_dataloader())