import numpy as np

from utils import load_mappings


mappings = load_mappings()

use_the_gpu = False

senttag2word = {'POS': 'positive', 'NEG': 'negative', 'NEU': 'neutral'}
enttag2opinions = {'POS': 'great', 'NEG': 'bad', 'NEU': 'ok'}
sentword2opinion = {'positive': 'great', 'negative': 'bad', 'neutral': 'ok'}

laptop_parent_mapping = mappings['laptop_parent_mapping']
laptop_full_mapping = mappings['laptop_full_mapping']
rest_full_mapping = mappings['restaurant_full_mapping']

laptop_dict = {}
for elt in laptop_parent_mapping:
    laptop_dict[elt[1]] = elt[0]

laptop_parent_dict = {}
for elt in laptop_parent_mapping:
    laptop_parent_dict[elt[1]] = elt[0]

laptop_full_dict = {}
for elt in laptop_full_mapping:
    laptop_full_dict[elt[1]] = elt[0]

rest_full_dict = {}
for elt in rest_full_mapping:
    rest_full_dict[elt[1]] = elt[0]

domain_map = {'restaurant': rest_full_dict,
              'laptop': laptop_full_dict,
              'laptop_parent': laptop_parent_dict
              }

domain_map_inverted = {'restaurant': rest_full_mapping,
                       'laptop': laptop_full_mapping,
                       'laptop_parent': laptop_parent_mapping
                       }


def get_domain(label):
    for key in domain_map:
        if label in domain_map[key]:
            return key
    assert False, "invalid domain"


def ex_contains_implicit_opinion(quads):
    return any([quad[3] == 'NULL' and quad[0] != 'NULL' for quad in quads])


def ex_contains_implicit_aspect(quads):
    return any([quad[0] == 'NULL' and quad[3] != 'NULL' for quad in quads])


def ex_contains_full_implicit(quads):
    return any([quad[0] == 'NULL' and quad[3] == 'NULL' for quad in quads])


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll))
    return results


def get_pos_vec_bert(quad, sent):
    current_spans = {quad[0], quad[3]}  # aspect and opinion terms
    zeroes_vec = np.zeros(len(sent))
    for ent in current_spans:
        ent_list = ent.split(" ")
        # find locations of explicit terms to focus on
        curr_indices = find_sub_list(ent_list, sent)
        if curr_indices:
            first_result = curr_indices[0]
            for idx in range(first_result[0], first_result[1]):
                zeroes_vec[idx] = 1
    return zeroes_vec


def get_gdp_data(sents, labels, task, truncated=False):
    def inner_fn(sent, label):
        aspect_terms = set()
        for quad in label:
            aspect_terms.update([quad[0], quad[3]])
        utt_str = " ".join(sent)
        indices = {}
        for term in aspect_terms:
            if term != 'NULL':
                indices[term] = utt_str.find(term)
            else:
                indices[term] = len(utt_str) + 1

        if task == 'gdp_wo_sorting':
            sorted_aspects = aspect_terms
        else:
            sorted_aspects = sorted(aspect_terms, key=indices.get)

        outputs = []
        seen_aspects = set()
        covered = set()

        for aspect in sorted_aspects:
            seen_aspects.add(aspect)
            for quad in label:
                if quad[0] in seen_aspects and quad[3] in seen_aspects and tuple(quad) not in covered:
                    if len(quad) == 4:
                        at, raw_ac, sp, ot = quad
                    if truncated == True:
                        raw_ac = raw_ac.split("#")[0]

                    domain = get_domain(raw_ac)
                    ac = domain_map[domain].get(raw_ac, raw_ac)

                    covered.add(tuple(quad))
                    man_ot = sentword2opinion[sp]  # 'POS' -> 'good'
                    if at == 'NULL':  # for implicit aspect term
                        at = 'it'

                    revised_quad = [at, ac, man_ot, ot]

                    pos_vec = get_pos_vec_bert(revised_quad, sent)
                    # this is the template used in our proposed GEN-SCL-NAT
                    # one_quad_sent = [ac, "| THE", at, "IS", ot, "|", sp]
                    one_quad_sent = ["THE", at, "IS", ot, "|", ac, "|", sp]

                    # for the abalation where we sort things, use nat categories, but use the original asqp format
                    if task == 'gdp_wo_intra':
                        one_quad_sent = [ac, 'is', man_ot, 'because', at, 'is', ot]

                    outputs.append((one_quad_sent, pos_vec))

        sent_len = len(sent)
        if task == 'gdp_wo_sorting':
            sorted_outputs = outputs
        else:
            sorted_outputs = sorted(outputs, key=lambda x: (
                max(loc for loc, val in enumerate(x[1]) if val == 1) if 1 in x[1] else sent_len, x[0]))
        total_sent = []
        total_pos = None
        total_neg = None

        for idx, output in enumerate(sorted_outputs):
            total_sent += output[0]

            if task not in ['gdp']:
                print(task)
                print('NOT SUPPORTED')
                import pdb
                pdb.set_trace()

            if idx != len(sorted_outputs) - 1:
                # add SSEP token
                total_sent.append('[SSEP]')

        return sent.copy(), total_sent

    sents, outputs = list(
        zip(*[inner_fn(sents[idx], labels[idx]) for idx in range(len(sents))]
            )
    )

    return sents, outputs
