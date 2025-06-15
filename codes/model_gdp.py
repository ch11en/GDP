import math
import warnings
import json
import losses
from pathlib import Path

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.modeling_outputs import (BaseModelOutput, Seq2SeqLMOutput)

from data_utils import get_dataset
from eval_utils import compute_scores
from losses import SupConLoss

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(768, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, attention_mask):
        last_state    = torch.mul(x, attention_mask.unsqueeze(-1))
        features      = torch.sum(last_state, dim=1)
        features_drop = self.dropout(features)
        return torch.stack((self.layer_1(features), self.layer_1(features_drop)), 1)


class GridLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_e = nn.Linear(768, 1024)
        self.layer_d = nn.Linear(768, 1024)
        self.layer_1 = nn.Linear(2048, 3)
        self.dropout = nn.Dropout(0.1)
        self.grid_loss_fct = CrossEntropyLoss(ignore_index=-100)

    def forward(self, encoder_last_hidden_state, decoder_hidden_states, encoder_mask, decoder_mask, grid_label):
        e_s_len = encoder_last_hidden_state.shape[1]
        d_s_len = decoder_hidden_states.shape[1]

        decoder_hidden_state = decoder_hidden_states[-1]

        encoder_state = torch.nn.functional.relu(self.layer_e(encoder_last_hidden_state))
        decoder_state = torch.nn.functional.relu(self.layer_d(decoder_hidden_state))

        encode_last_state = torch.mul(encoder_state, encoder_mask.unsqueeze(-1))
        decode_last_state = torch.mul(decoder_state, decoder_mask.unsqueeze(-1))

        encode_last_state = encode_last_state.unsqueeze(1).expand(-1, d_s_len, -1, -1)
        decode_last_state = decode_last_state.unsqueeze(2).expand(-1, -1, e_s_len, -1)

        grid_state = torch.cat([decode_last_state, encode_last_state], dim=3)
        grid_state = self.dropout(grid_state)

        features_summed = self.layer_1(grid_state)

        dropped   = self.dropout(features_summed)
        grid_loss = self.grid_loss_fct(dropped.view(-1, dropped.size(-1)), grid_label.view(-1))

        return grid_loss


class LightningModule(pl.LightningModule):

    def __init__(self, args, model_name_or_path, tokenizer):
        super(LightningModule, self).__init__()

        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.test_result = None
        self.save_hyperparameters(ignore=['grid_model', 'sp_model', 'ot_model', 'at_model', 'ac_model'])

        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.tokenizer = tokenizer

        self.dropout = nn.Dropout(args.p)

        self.df_num = args.df_num

        self.grid_model = GridLinearModel()
        self.sp_model = LinearModel()
        self.ot_model = LinearModel()
        self.at_model = LinearModel()
        self.ac_model = LinearModel()

    def is_logger(self):
        return True

    def vagueify(self, sequence_output):
        all_logits = None
        for _ in range(self.df_num):
            temp = self.dropout(sequence_output)

            if self.model.config.tie_word_embeddings:
                temp = temp * (self.model.model_dim ** -0.5)

            lm_logits = self.model.lm_head(temp).unsqueeze(0)

            if all_logits is not None:
                all_logits = torch.cat((all_logits, lm_logits), dim=0)
            else:
                all_logits = lm_logits
        return all_logits

    @torch.no_grad()
    def get_vague_samples(self, all_logits, labels):
        all_top_k_ids = torch.topk(all_logits, 1, dim=-1)
        all_c = all_top_k_ids.indices.squeeze(-1)

        batch_size = labels.shape[0]
        all_lengths = (labels != 0).sum(-1)
        mask_results = torch.zeros_like(all_logits)

        updated_labels = labels
        old_value = -100
        new_value = self.args.pad_value
        updated_labels[updated_labels == old_value] = new_value

        for i in range(all_c.shape[0]):
            for j in range(batch_size):
                mask_results[i, j, :all_lengths[j]] = mask_results[i, j, :all_lengths[j]].scatter(
                    -1, all_c[i, j, :all_lengths[j]].unsqueeze(-1), 1)
            mask_results[i] = mask_results[i].scatter(-1, updated_labels.unsqueeze(-1), 0)

        return mask_results

    def compute_vague_loss(self, all_logits, N_mask_results, lm_labels):
        log_softmax = nn.LogSoftmax(dim=-1)
        softmax_fct = nn.Softmax(dim=-1)

        label_masks = torch.ones_like(lm_labels)
        label_masks[lm_labels == 0] = 0
        softmax_logits = softmax_fct(all_logits)
        all_log_softmax_logits = log_softmax(all_logits)

        df_num = softmax_logits.shape[0]
        n_logits = (self.args.gama * softmax_logits).exp() * N_mask_results
        n_logits = n_logits.sum(0).sum(-1)

        labels = lm_labels.unsqueeze(0).repeat(df_num, 1, 1).unsqueeze(-1)
        p_logits = torch.gather((- (self.args.gama * softmax_logits)).exp(), -1, labels)
        p_logits = p_logits.sum(0).squeeze(-1)
        label_loss = torch.log(1 + math.exp(self.args.m * self.args.gama) * n_logits * p_logits) * label_masks
        label_loss = label_loss.sum()

        vocab_size = all_log_softmax_logits.shape[-1]
        loss_fct = nn.NLLLoss(ignore_index=-100, reduction='sum')
        all_likelihood_loss = torch.stack([
            loss_fct(logits.reshape(-1, vocab_size), lm_labels.view(-1))
            for logits in all_log_softmax_logits
        ])
        likelihood_loss = torch.mean(all_likelihood_loss)

        regular_loss = softmax_logits * all_log_softmax_logits
        mi_loss = -((regular_loss * label_masks.unsqueeze(0).unsqueeze(-1)).sum())

        return label_loss + likelihood_loss + mi_loss

    def train_dataloader(self):
        self.train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.args)
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, drop_last=True, shuffle=True, num_workers=4)

    def val_dataloader(self):
        self.val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.args)
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=4)

    def test_dataloader(self):
        self.test_dataset = get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.args)
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=4)

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):

        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        encoder_last_hidden_states = encoder_outputs.last_hidden_state

        if labels is not None and decoder_input_ids is None:
            decoder_input_ids = self.model._shift_right(labels)

        decoder_outputs = self.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_last_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=True
        )
        decoder_last_layer_hiddens = decoder_outputs.last_hidden_state

        all_logits     = self.vagueify(decoder_last_layer_hiddens)
        N_mask_results = self.get_vague_samples(all_logits, labels)
        loss           = self.compute_vague_loss(all_logits, N_mask_results, labels)

        main_output = Seq2SeqLMOutput(loss=loss, logits=all_logits)

        if self.sp_model or self.ot_model or self.at_model:
            masked_encoder = torch.mul(encoder_last_hidden_states, attention_mask.unsqueeze(-1))
            pooled_encoder = normalize(torch.sum(masked_encoder, dim=1), p=2.0, dim=1)

            sp_pred = self.sp_model(encoder_last_hidden_states, attention_mask) if self.sp_model else None
            ot_pred = self.ot_model(encoder_last_hidden_states, attention_mask) if self.ot_model else None
            at_pred = self.at_model(encoder_last_hidden_states, attention_mask) if self.at_model else None

            return main_output, encoder_last_hidden_states, decoder_last_layer_hiddens, sp_pred, ot_pred, at_pred, pooled_encoder
        else:
            return main_output, encoder_last_hidden_states, decoder_last_layer_hiddens

    def _step(self, batch):
        lm_labels = batch["target_ids"].clone()
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100

        if self.current_epoch < self.args.stat_full_train_ep:
            ac_mask_matrix = batch["ac_mask_matrix"]
            lm_labels = lm_labels * ac_mask_matrix + ((1 - ac_mask_matrix) * -100)

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0].loss

        if getattr(self.args, 'category_loss_lambda', 0) > 0:
            ac_mask_matrix = batch["ac_mask_matrix"]
            category_labels = lm_labels * ac_mask_matrix + ((1 - ac_mask_matrix) * -100)
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss += loss_fct(outputs[1].view(-1, outputs[1].size(-1)),
                             category_labels.view(-1)) * self.args.category_loss_lambda

        # Add contrastive losses if enabled
        if getattr(self.args, 'scl_loss_lambda', 0) > 0 and len(outputs) > 3:
            criterion = SupConLoss(loss_scaling_factor=self.args.cont_loss,
                                   temperature=self.args.cont_temp)

            sp_loss = criterion(normalize(outputs[3], p=2.0, dim=2), batch['sp_labels']) * self.args.scl_loss_lambda
            at_loss = criterion(normalize(outputs[4], p=2.0, dim=2), batch['at_labels']) * self.args.scl_loss_lambda
            ot_loss = criterion(normalize(outputs[5], p=2.0, dim=2), batch['ot_labels']) * self.args.scl_loss_lambda
            loss += sp_loss + at_loss + ot_loss

        if getattr(self.args, 'grid_loss_lambda', 0) > 0 and len(outputs) > 1:
            grid_loss = self.grid_model(
                outputs[1],  # encoder_last_state
                outputs[2],  # decoder_hidden_states
                batch["source_mask"],
                batch["target_mask"],
                batch['grid_label']
            ) * self.args.grid_loss_lambda
            loss += grid_loss

        return loss, outputs

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):

        outs = self.model.generate(
            input_ids=batch['source_ids'],
            attention_mask=batch['source_mask'],
            max_length=self.args.max_seq_length * 2,
            num_beams=5
        )

        out_dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        tar_dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]

        self.test_result  = {'outputs': out_dec, 'targets': tar_dec}

        return self.test_result

    def on_test_epoch_end(self):

        outputs = self.test_result['outputs']
        targets = self.test_result['targets']

        scores, all_labels, all_preds = compute_scores(outputs, targets, self.args.task, False)

        results = {
            'labels_correct': all_labels,
            'labels_pred': all_preds,
            'output_pred': outputs,
            'output_correct': targets,
        }

        ex_list = []
        for idx in range(len(all_preds)):
            new_dict = {}
            for key in results:
                new_dict[key] = results[key][idx]
            ex_list.append(new_dict)

        results = {'performance_metrics': scores, 'examples': ex_list}

        json.dump(results, open(f"{self.args.output_dir}/results-{self.args.dataset}.json", 'w'), indent=2, sort_keys=True)

        print(scores)

    def get_training_steps(self):
        dataloader = self.train_dataloader()
        total_steps = (len(dataloader.dataset) // (self.args.batch_size * max(1, len(self.args.gpus))))

        total_steps = total_steps // self.args.gradient_accumulation_steps * float(self.args.num_train_epochs)

        return total_steps


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in self.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(params, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        t_total = self.get_training_steps()
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
