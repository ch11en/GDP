import parse
import torch
from tqdm import tqdm
import json

sentiment_word_list = ['positive', 'negative', 'neutral']
opinion2word = {'great': 'positive', 'bad': 'negative', 'ok': 'neutral'}
opinion2word_under_o2m = {'good': 'positive', 'great': 'positive', 'best': 'positive',
                          'bad': 'negative', 'okay': 'neutral', 'ok': 'neutral', 'average': 'neutral'}

numopinion2word = {'SP1': 'positive', 'SP2': 'negative', 'SP3': 'neutral'}


def extract_spans_para(task, seq, seq_type):
    quads = []
    sents = [s.strip() for s in seq.split('[SSEP]')]

    if task == 'asqp' or task == 'gdp_wo_intra':
        for s in sents:
            # example: food quality is bad because pizza is over cooked.
            try:
                ac_sp, at_ot = s.split(' because ')
                ac, sp = ac_sp.split(' is ')
                at, ot = at_ot.split(' is ')

                # if the aspect term is implicit
                if at.lower() == 'it':
                    at = 'NULL'
            except ValueError:
                try:
                    pass
                except UnicodeEncodeError:
                    pass
                try:
                    result = list(parse.parse('{0} is {1} because {2} is{3}', s, case_sensitive=True))
                    ac, sp, at, ot = result[0], result[1], result[2], result[3].lstrip(' ')
                except:
                    print(s)
                    ac, at, sp, ot = '', '', '', ''
            quads.append((ac, at, sp, ot))

    elif task.startswith('gdp'):
        if seq:
            for s in sents:
                # food quality|bad|pizza|over cooked.
                try:
                    result = list(parse.parse('{0}|{1}|{2}|{3}', s, case_sensitive=True))
                    ac, at, ot, sp = [elt.strip(' ') for elt in result]
                    if at.lower() == 'it':
                        at = 'NULL'
                except:
                    try:
                        # The food qualityisbad|pizza|over cooked.
                        result = list(parse.parse('THE{0}IS{1}|{2}|{3}', s, case_sensitive=True))
                        at, ot, ac, sp = [elt.strip(' ') for elt in result]
                        if at.lower() == 'it':
                            at = 'NULL'
                    except:
                        print(s)
                        ac = ''
                        sp = ''
                        at = 'NULL'
                        ot = 'NULL'

                quads.append((ac, at, sp, ot))

    else:
        raise NotImplementedError
    return quads


def compute_ao_pair_f1_scores(pred_pt, gold_pt, silent=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        pred_pt_pair_set = set((pp[1], pp[3]) for pp in pred_pt[i])
        gold_pt_pair_set = set((gp[1], gp[3]) for gp in gold_pt[i])
        n_gold += len(pred_pt_pair_set)
        n_pred += len(gold_pt_pair_set)
        for t in pred_pt_pair_set:
            if t in gold_pt_pair_set:
                n_tp += 1

    if not silent:
        print(f"number of gold pairs: {n_gold}, predicted pairs: {n_pred}, hit: {n_tp}")

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    if recall > 1.0:
        import pdb
        pdb.set_trace()
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores

def compute_f1_scores(pred_pt, gold_pt, silent=True):
    """
    Function to compute F1 scores with pred and gold quads
    The input needs to be already processed
    """
    # number of true postive, gold standard, predictions
    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        n_gold += len(set(gold_pt[i]))
        n_pred += len(set(pred_pt[i]))

        for t in set(pred_pt[i]):
            if t in gold_pt[i]:
                n_tp += 1

    if not silent:
        print(f"number of gold spans: {n_gold}, predicted spans: {n_pred}, hit: {n_tp}")
    
    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    if recall > 1.0:
        import pdb
        pdb.set_trace()
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'precision': precision, 'recall': recall, 'f1': f1}

    return scores


def split_imp_exp_instances(instances):
    ee_instances, ei_instances, ie_instances, ii_instances = [], [], [], []
    for instance in instances:
        ac, at, sp, ot = instance
        if at == "NULL" and ot == "NULL":
            ii_instances.append(instance)
        elif at == "NULL" and ot != "NULL":
            ie_instances.append(instance)
        elif at != "NULL" and ot == "NULL":
            ei_instances.append(instance)
        else:
            ee_instances.append(instance)
    return ee_instances, ei_instances, ie_instances, ii_instances

def compute_scores(pred_seqs, gold_seqs, task, silent=True):
    """
    Compute model performance
    """
    assert len(pred_seqs) == len(gold_seqs)
    num_samples = len(gold_seqs)

    all_labels, all_preds = [], []
    ee_all_labels, ei_all_labels, ie_all_labels, ii_all_labels = [], [], [], []
    ee_all_preds, ei_all_preds, ie_all_preds, ii_all_preds = [], [], [], []

    for i in range(num_samples):
        gold_list = extract_spans_para(task, gold_seqs[i], 'gold')
        pred_list = extract_spans_para(task, pred_seqs[i], 'pred')

        ee_gold, ei_gold, ie_gold, ii_gold = split_imp_exp_instances(gold_list)
        ee_pred, ei_pred, ie_pred, ii_pred = split_imp_exp_instances(pred_list)

        all_labels.append(gold_list)
        all_preds.append(pred_list)

        ee_all_labels.append(ee_gold)
        ei_all_labels.append(ei_gold)
        ie_all_labels.append(ie_gold)
        ii_all_labels.append(ii_gold)

        ee_all_preds.append(ee_pred)
        ei_all_preds.append(ei_pred)
        ie_all_preds.append(ie_pred)
        ii_all_preds.append(ii_pred)

    if not silent:
        print("\nResults:")
        scores = compute_f1_scores(all_preds, all_labels, silent)
        # pair_scores = compute_ao_pair_f1_scores(all_preds, all_labels, silent)
        print(scores)
        # print(pair_scores)
        ee_scores = compute_f1_scores(ee_all_preds, ee_all_labels, silent)
        ei_scores = compute_f1_scores(ei_all_preds, ei_all_labels, silent)
        ie_scores = compute_f1_scores(ie_all_preds, ie_all_labels, silent)
        ii_scores = compute_f1_scores(ii_all_preds, ii_all_labels, silent)
        print("ee: ", ee_scores)
        print("ei: ", ei_scores)
        print("ie: ", ie_scores)
        print("ii: ", ii_scores)
    else:
        scores = compute_f1_scores(all_preds, all_labels, silent)

    return scores, all_labels, all_preds