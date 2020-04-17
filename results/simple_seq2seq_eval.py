import json, os, errno
import argparse
from collections import defaultdict
from tabulate import tabulate

SRL_ARGS = ["A1", "AM-LOC", "AM-MNR", "A2", "A0", "C-A1", "AM-TMP", "A3", "A4", "AM-MOD",
            "R-A0", "AM-NEG", "AM-EXT", "AM-DIS", "AM-ADV", "R-AM-TMP", "AM-PNC", "R-A1", "AM-DIR", "R-A2",
            "R-AM-PNC", "C-AM-MNR", "AM-PRD", "R-A3", "AM-CAU", "R-AM-MNR", "R-AM-LOC", "C-AM-TMP", "C-A2",
            "C-A0", "R-AM-EXT", "A5)", "C-AM-LOC", "C-AM-ADV", "C-A3", "R-AM-CAU", "R-A4", "R-AM-ADV",
            "AM-TM", "AM", "AM-REC", "C-AM-PNC", "C-AM-DIR", "AM-PRT", "C-AM-EXT", "C-A4", "AA", "R-AA",
            "C-AM-DIS", "C-AM-NEG", "C-R-AM-TMP", "C-AM-CAU", "R-AM-DIR"]

SKIP_TOKENS = ["(#", "V)"]

AVAILABLE_LANGS = {"<EN>": 0, "<EN-SRL>": 1,
                    "<DE>": 2, "<DE-SRL>": 3,
                    "<FR>": 4, "<FR-SRL>": 5,
                    "<ES>": 6, "<ES-SRL>": 7}

SPECIAL_TOKENS = ["@END@"] + list(AVAILABLE_LANGS.keys())


def create_directory(filepath):
    try:
        os.makedirs(filepath)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _sequence_is_balanced(seq, all_arguments_dict):
    open_args, close_args = 0, 0
    for tok in seq:
        tok_upp = tok.upper().strip(")")
        if "(#" == tok_upp:
            open_args += 1
        elif tok_upp in all_arguments_dict or tok_upp == "V":
            close_args += 1
    return open_args == close_args


def _sequence_has_duplicates(sys_labelset):
    predicted_unique_args = []
    for arg_obj in sys_labelset:
        if arg_obj["tag"] not in predicted_unique_args:
            predicted_unique_args.append(arg_obj["tag"])
    return len(predicted_unique_args) < len(sys_labelset)


def get_metrics(false_pos, false_neg, true_pos):
    _denom = true_pos + false_pos
    precision = true_pos / _denom if _denom else 0
    _denom = true_pos + false_neg
    recall = true_pos / _denom if _denom else 0
    _denom = precision + recall
    F1 = 2 * ((precision * recall) / _denom) if _denom else 0
    return precision*100, recall*100, F1*100


def print_overall_metrics(arg_excess, arg_missed, arg_match):
    processed_args = set()
    results = []
    tot_excess, tot_missed, tot_match = 0, 0, 0
    for arg, count in arg_match.items():
        excess = arg_excess.get(arg, 0)
        missed = arg_missed.get(arg, 0)
        p,r,f = get_metrics(false_pos=excess, false_neg=missed, true_pos=count)
        processed_args.add(arg)
        results.append((arg, count, excess, missed, p, r, f))
        tot_excess += excess
        tot_missed += missed
        tot_match += count
    for arg, count in arg_excess.items():
        if arg not in processed_args:
            excess = count
            missed = arg_missed.get(arg, 0)
            correct = arg_match.get(arg, 0)
            p, r, f = get_metrics(false_pos=excess, false_neg=missed, true_pos=correct) # p,r,f = 0,0,0
            processed_args.add(arg)
            results.append((arg, correct, excess, missed, p, r, f))
            tot_excess += excess
            tot_missed += missed
            tot_match += correct
    for arg, count in arg_missed.items():
        if arg not in processed_args:
            excess = arg_excess.get(arg, 0)
            correct = arg_match.get(arg, 0)
            missed = count
            p, r, f = get_metrics(false_pos=excess, false_neg=missed, true_pos=correct) # p,r,f = 0,0,0
            results.append((arg, correct, excess, missed, p, r, f))
            tot_excess += excess
            tot_missed += missed
            tot_match += correct
    results = sorted(results, key= lambda x: x[0])

    prec, rec, F1 = get_metrics(false_pos=tot_excess, false_neg=tot_missed, true_pos=tot_match)

    print("\n--- OVERALL ---\nCorrect: {0}\tExcess: {1}\tMissed: {2}\nPrecision: {3:.2f}\t\tRecall: {4:.2f}\nF1: {5:.2f}\n".format(tot_match, tot_excess, tot_missed, prec, rec, F1))
    print(tabulate(results, headers=["corr.", "excess", "missed", "prec.", "rec.", "F1"], floatfmt=".2f"))


def get_sentence_args(tokenlist, consider_verb_label, all_arguments_dict):
    pred_args = []
    pix, predicate = -1, "<NONE>"
    words_only, bio_seq = [], []
    tag = "O"

    for i, tok in enumerate(tokenlist):
        tok_upp = tok.upper().strip(")")
        if tok_upp == "V":
            predicate = tokenlist[i - 1]
            if consider_verb_label: pred_args.append({"tag": tok_upp, "head": predicate.lower()})
            pix = i
            tag = "B-V"
        elif tok_upp in all_arguments_dict:
            pred_args.append({"tag": tok_upp, "head": tokenlist[i-1].lower()})
            tag = "B-"+tok_upp
        elif tok_upp != "(#":
            words_only.append(tok.lower())
            bio_seq.append(tag)
            tag = "O"
    # Manual correction to aviod shift in BIO (it happens because tag is after the word and not before)
    bio_seq = bio_seq[1:] + ["O"]
    return (pix, predicate), pred_args, words_only, bio_seq


def evaluate_tagset(gld, sys, gld_pred, sys_pred, consider_verb_token, consider_verb_position):
    gld_ix, gld_pred = gld_pred
    sys_ix, sys_pred = sys_pred
    if consider_verb_token:
        if gld_pred == "<NONE>" and "V" in sys:  # Count all tags as EXCESS
            print("MisMatch!", gld_pred, sys_pred)
            return {"excess": [x.split("_")[0] for x in sys], "missed": [], "match": []}
        elif gld_pred.lower() != sys_pred.lower():  # Predicate Missmatch: Count all tags as MISSING, Including Predicate!
            return {"excess": [], "missed": [x.split("_")[0] for x in sys] + ["V"], "match": []}

    if consider_verb_position and gld_ix != sys_ix: # Predicate Missmatch: Count all tags as MISSING, Including Predicate!
        return {"excess": [], "missed": [x.split("_")[0] for x in sys] + ["V"], "match": []}

    excess = sys - gld  # False Positives
    missed = gld - sys  # False Negatives
    true_pos = sys.intersection(gld)

    eval_obj = {"excess": [x.split("_")[0] for x in excess],
                "missed": [x.split("_")[0] for x in missed],
                "match": [x.split("_")[0] for x in true_pos]}
    return eval_obj


def simple_output_analysis(instances, metadata, consider_verb_label, consider_verb_token, consider_verb_position):
    verb_mapping, full_labelset_mapping = defaultdict(list), defaultdict(list)
    confusion_dict = defaultdict(list)
    all_excess, all_missed, all_match = defaultdict(int), defaultdict(int), defaultdict(int)

    unequal_outputs, unbalanced_sequence, duplicate_args_sequence = 0, 0, 0

    def _add_to_eval_dicts(eval_metrics, arg_excess, arg_missed, arg_match):
        for arg in eval_metrics["excess"]:
            arg_excess[arg] += 1
        for arg in eval_metrics["missed"]:
            arg_missed[arg] += 1
        for arg in eval_metrics["match"]:
            arg_match[arg] += 1

    def get_arg_struct_mapping(exp_pred, exp_args, tgt_pred, tgt_args):
        """
        :param exp_pred: (6, 'say')
        :param exp_args: [{'tag': 'A1', 'head': 'Frau'}, {'tag': 'A0', 'head': 'ich'}]
        :param tgt_pred: (9, 'sagen')
        :param tgt_args: [{'tag': 'A0', 'head': 'i'}, {'tag': 'AM-MOD', 'head': 'would'}, {'tag': 'A1', 'head': 'to'}]
        :return:
        """
        verb_mapping[exp_pred[1]].append(tgt_pred[1])
        exp_arg_lbl = [f"{x['tag']}_{x['head']}" for x in exp_args]
        tgt_arg_lbl = [f"{y['tag']}_{y['head']}"  for y in tgt_args]
        eval_metrics = evaluate_tagset(set(exp_arg_lbl), set(tgt_arg_lbl), exp_pred, tgt_pred,
                                       consider_verb_token=consider_verb_token,
                                       consider_verb_position=consider_verb_position)
        full_labelset_mapping[tuple(exp_arg_lbl)].append(tuple(tgt_arg_lbl))
        pairwise_tags = []
        for i, x in enumerate(exp_arg_lbl):
            if i < len(tgt_arg_lbl):
                pairwise_tags.append((x, tgt_arg_lbl[i]))
        for gld_label, sys_label in pairwise_tags:
            confusion_dict[gld_label].append(sys_label)
        return eval_metrics

    # Iterate through Test Data ...
    for (src, tgt), meta in zip(instances, metadata):
        expected_tokens = meta.get("original_target", None)

        if expected_tokens:
            # Get Info from System Output (Target)
            predicted_tokens = tgt
            tgt_predicate_word, tgt_args, tgt_words, tgt_bio = get_sentence_args(predicted_tokens, consider_verb_label, SRL_ARGS)
            exp_predicate_word, exp_args, exp_words, _ = get_sentence_args(expected_tokens, consider_verb_label, SRL_ARGS)

            # We track Mal-formations, but still evaluate for Pred-Arg Performance
            if len(tgt_words) != len(exp_words):
                unequal_outputs += 1
            elif not _sequence_is_balanced(predicted_tokens, SRL_ARGS):
                unbalanced_sequence += 1
            elif _sequence_has_duplicates(tgt_args):
                duplicate_args_sequence += 1

            # Evaluation Metrics ALL
            eval_metrics_all = get_arg_struct_mapping(exp_predicate_word, exp_args, tgt_predicate_word, tgt_args)
            _add_to_eval_dicts(eval_metrics_all, all_excess, all_missed, all_match)

        else:
            print("MetaData does not include original_target key, therefore argument analysis is skept!")
            return

    # Overall Metrics
    print("\n\n------------ OVERALL Evaluation ------------\n")
    print("\n{} sequences had different length!".format(unequal_outputs))
    print("{} contained duplicate arguments!".format(duplicate_args_sequence))
    print("{} were Unbalanced Bracketing!\n".format(unbalanced_sequence))
    print_overall_metrics(all_excess, all_missed, all_match)


def read_srl_json_output(filename):
    def _trim_source(src):
        trimmed = []
        for tok in src:
            if tok.upper() not in SPECIAL_TOKENS:
                trimmed.append(tok)
        return trimmed

    def get_src_tgt(json_obj):
        # JSON : ['loss', 'metadata', 'predicted_log_probs', 'predictions', 'predicted_tokens']
        # JSON METADATA : ['source_tokens', 'target_tokens', 'verb', 'src_lang', 'tgt_lang', 'original_BIO',
        #                  'original_predicate_senses', 'predicate_senses', 'original_target']
        src = json_obj["metadata"]["source_tokens"]
        tgt = json_obj["predicted_tokens"][0]
        meta = json_obj["metadata"]
        src = _trim_source(src)
        return src, tgt, meta
    # Read lines from the JSON files
    gold, instances, metadata = [], [], []
    with open(filename) as f:
        for line in f:
            s, t, m = get_src_tgt(json.loads(line))
            instances.append((s, t))
            metadata.append(m)
    return instances, metadata


if __name__ == "__main__":
    """
    RUN EXAMPLE: 
        python results/simple_seq2seq_eval.py --sys_out results/copynet-srl-conll09-bert_english_test.json
    """

    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sys_out', help='JSON File with the model outputs', required=True)
    args = parser.parse_args()

    # Process the instances
    sys_instances, sys_meta = read_srl_json_output(args.sys_out)

    # Evaluate Outputs
    simple_output_analysis(sys_instances, sys_meta,
                           consider_verb_label=False, # Include (or not) Predicates in F1 Score
                           consider_verb_token=True, # Count as correct ONLY IF token in SRC and TGT are the same
                           consider_verb_position=False) # Count as correct ONLY IF token in SRC and TGT are in the same position
