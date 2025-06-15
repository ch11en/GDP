import random
from torch.utils.data import Dataset
import torch
from generate_data import get_gdp_data, get_domain, domain_map

sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}


def get_dataset(tokenizer, type_path, args):
    return GDPDATASET(tokenizer=tokenizer,
                      data_dir=args.dataset,
                      data_type=type_path,
                      max_len=args.max_seq_length,
                      task=args.task,
                      truncate=args.truncate)


def read_line_examples_from_file(data_path, silence=False):
    """
        Each line is: sent####labels
        Return List[List[word]], List[Tuple]
    """
    sentences, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        words, labels = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                words, tuples = line.split('####')
                sentences.append(words.split())
                labels.append(eval(tuples))
    return sentences, labels


def get_para_asqp_targets(sents, labels, truncated=False):
    targets = []
    for label in labels:
        all_quad_sentences = []
        for quad in label:
            at, ac, sp, ot = quad

            if truncated:
                ac = ac.split("#")[0]
            man_ot = sentword2opinion[sp]
            if at == 'NULL':  # for implicit aspect term
                at = 'it'

            one_quad_sentence = f"{ac} is {man_ot} because {at} is {ot}"
            all_quad_sentences.append(one_quad_sentence)

        target = ' [SSEP] '.join(all_quad_sentences)
        targets.append(target)
    return sents.copy(), targets


def replace_unk_tokens(string):
    replace_dict = {"`": "'", }
    for pstr in replace_dict:
        string = string.replace(pstr, replace_dict[pstr])
    return string


def get_transformed_io(data_path, data_dir, task, data_type, truncate=False):
    sentences, labels = read_line_examples_from_file(data_path)
    inputs, targets = get_gdp_data(sentences, labels, task, truncate)
    return inputs, targets, labels


class GDPDATASET(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, task, max_len=256, data_path=None, truncate=False,
                 with_labels=True):
        if data_path:
            self.data_path = data_path
        else:
            self.data_path = f'../data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.task = task
        self.data_type = data_type
        self.inputs = []
        self.targets = []
        self.sentence_strings = []
        self.truncate = truncate
        self.with_labels = with_labels

        if self.with_labels:
            self.contrastive_labels = {
                'sp': [],
                'at': [],
                'ot': []
            }
            self.grid_labels = []
            self.category_mask_matrixes = []

        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids  = self.inputs[index]["input_ids"].squeeze()
        src_mask    = self.inputs[index]["attention_mask"].squeeze()

        target_ids  = self.targets[index]["input_ids"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()

        item = {
            "source_ids": source_ids,
            "source_mask": src_mask,
            "target_ids": target_ids,
            "target_mask": target_mask
        }

        if self.with_labels:
            item.update({
                'sp_labels': torch.tensor(self.contrastive_labels['sp'][index]),
                'ot_labels': torch.tensor(self.contrastive_labels['ot'][index]),
                'at_labels': torch.tensor(self.contrastive_labels['at'][index]),
                'grid_label': self.grid_labels[index],
                'ac_mask_matrix': self.category_mask_matrixes[index]
            })

        return item

    def get_raw_labels(self):
        results = get_transformed_io(self.data_path, self.data_dir, self.task, self.data_type, self.truncate)
        return results

    def _build_examples(self):
        inputs, targets, labels = get_transformed_io(
            self.data_path, self.data_dir, self.task, self.data_type, self.truncate
        )
        self.sentence_strings = inputs

        for i in range(len(inputs)):
            input_text = ' '.join(inputs[i])
            input_text = replace_unk_tokens(input_text)

            target_text = targets[i]

            if isinstance(targets[i], list):
                target_text = " ".join(targets[i])
                target_text = replace_unk_tokens(target_text)

            tokenized_input = self.tokenizer.batch_encode_plus(
                [input_text], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )
            tokenized_target = self.tokenizer.batch_encode_plus(
                [target_text], max_length=self.max_len, padding="max_length",
                truncation=True, return_tensors="pt"
            )

            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)

        if self.with_labels and labels:
            self._process_labels(labels)

    def _process_labels(self, labels):
        sentiment_dict = {'negative': 0, 'neutral': 1, 'positive': 2, 'mixed': 3}
        self.contrastive_labels['sp'] = [
            sentiment_dict[list(set([quad[2] for quad in ex]))[0]] if len(set([quad[2] for quad in ex])) == 1
            else sentiment_dict['mixed']
            for ex in labels
        ]

        opinion_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        self.contrastive_labels['ot'] = [
            opinion_dict['EXPLICIT'] if 'NULL' not in set([quad[3] for quad in ex])
            else (opinion_dict['NULL'] if len(set([quad[3] for quad in ex])) == 1
                  else opinion_dict['BOTH'])
            for ex in labels
        ]

        aspect_dict = {'NULL': 0, 'EXPLICIT': 1, 'BOTH': 2}
        self.contrastive_labels['at'] = [
            aspect_dict['EXPLICIT'] if 'NULL' not in set([quad[0] for quad in ex])
            else (aspect_dict['NULL'] if len(set([quad[0] for quad in ex])) == 1
                  else aspect_dict['BOTH'])
            for ex in labels
        ]

        self.grid_labels, self.category_mask_matrixes = self._build_grid_matrices(
            self.inputs, self.targets, labels
        )

    def _build_grid_matrices(self, inputs, targets, labels):
        grid_member_dict = {
            'pad': -100,
            'none': 0,
            'aspect': 1,
            'opinion': 2,
            'negative': 3,
            'neutral': 4,
            'positive': 5,
        }
        grid_matrixes = []
        category_mask_matrixes = []

        for i in range(len(inputs)):
            sentence_encoded = inputs[i]["input_ids"].squeeze()
            targets_encoded  = targets[i]["input_ids"].squeeze()

            non_pad_sentence_len = sum(self.tokenizer.pad_token_id != sentence_encoded).item()
            non_pad_target_len   = sum(self.tokenizer.pad_token_id != targets_encoded).item()

            grid_matrix = torch.full((len(targets_encoded), len(sentence_encoded)), grid_member_dict['pad'])
            category_mask_matrix = torch.zeros((len(targets_encoded),))

            sentence_map = self._build_token_to_char_map(sentence_encoded, non_pad_sentence_len)
            target_map   = self._build_token_to_char_map(targets_encoded, non_pad_target_len)

            for quad in labels[i]:
                at, ac, _, ot = quad
                if self.truncate:
                    ac = ac.split("#")[0]
                domain = get_domain(ac)
                asc = domain_map[domain].get(ac, ac)

                asc_clean = replace_unk_tokens(asc.replace(" ", ""))
                asc_pos = self._find_token_positions(asc_clean, targets_encoded, target_map, non_pad_target_len)
                if asc_pos:
                    grid_matrix[asc_pos[0]:asc_pos[1], :non_pad_sentence_len] = grid_member_dict['none']
                    category_mask_matrix[asc_pos[0]:asc_pos[1]] = 1

                if at != "NULL":
                    at_clean = replace_unk_tokens(at.replace(" ", ""))
                    at_pos = self._find_token_positions(at_clean, sentence_encoded, sentence_map, non_pad_sentence_len)
                    if at_pos and asc_pos:
                        grid_matrix[asc_pos[0]:asc_pos[1], at_pos[0]:at_pos[1]] = grid_member_dict['aspect']

                if ot != "NULL":
                    ot_clean = replace_unk_tokens(ot.replace(" ", ""))
                    ot_pos = self._find_token_positions(ot_clean, sentence_encoded, sentence_map, non_pad_sentence_len)
                    if ot_pos and asc_pos:
                        grid_matrix[asc_pos[0]:asc_pos[1], ot_pos[0]:ot_pos[1]] = grid_member_dict['opinion']

            grid_matrixes.append(grid_matrix)
            category_mask_matrixes.append(category_mask_matrix)

        return grid_matrixes, category_mask_matrixes

    def _build_token_to_char_map(self, tokens, non_pad_len):
        decoded_str = ""
        token_map = []
        for i in range(1, non_pad_len + 1):
            current_decoded = self.tokenizer.decode(tokens[:i], clean_up_tokenization_spaces=False).replace(' ', '')
            new_chars = len(current_decoded) - len(decoded_str)
            token_map.extend([i - 1] * new_chars)
            decoded_str = current_decoded
        return token_map

    def _find_token_positions(self, text, tokens, token_map, non_pad_len):
        full_text = self.tokenizer.decode(tokens[:non_pad_len], clean_up_tokenization_spaces=False).replace(' ', '')
        if text not in full_text:
            return None

        start = full_text.index(text)
        end = start + len(text) - 1
        return (token_map[start], token_map[end] + 1)