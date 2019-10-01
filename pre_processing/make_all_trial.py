"""
RUN EXAMPLE:
        python pre_processing/make_all_trial.py

Copy one sentence per predicate and save them in a JSON file.
Example:
    *) A CoNLL file contains 2 sentences, one with 2 predicates and one with 3 predicates.
    *) The final JSON file will contain up to 5 lines, the first 2 lines will have the same sentence
       but each copy will be labeled with only one predicate-argument structure,
       the other 3 lines will be copies of the seconde sentence.
    *) NOTE: In many cases the final JSON will contain less than this, because the Flair Tagger did not
        find a matching predicate that coincides with the labeled predicate on the target side.
"""


from CoNLL_Annotations import CoNLL05_Token, CoNLL05_Test_Token, CoNLL09_Token, CoNLL09_FrenchTestToken, read_conll
from flair.data import Sentence
from flair.models import SequenceTagger
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
import json, io


def make_one_pred_per_sent(predicate_json):
    all_preds = []
    with open(predicate_json) as f:
        for line in f:
            obj = json.loads(line)
            all_preds.append(obj["predicates"])
    return all_preds


def get_lang(lang):
    if lang =="<EN>":
        return spacy.load("en")
    elif lang == "<DE>":
        return spacy.load("de")
    elif lang == "<FR>":
        return spacy.load("fr")
    else:
        return None


def tokenize_postag_sentence(sentence_str, spacy_lang=None):
    if not spacy_lang:
        return sentence_str.split(), []
    else:
        toks, pos = zip(*[(tok.text, tok.pos_) for tok in spacy_lang(sentence_str)])
        return toks, pos


def get_source_frames(sentence_tokens, frame_tagger):
    all_frames = []
    if frame_tagger:
        sentence_obj = Sentence(" ".join(sentence_tokens))
        frame_tagger.predict(sentence_obj)
        for frame in sentence_obj.get_spans('frame'):
            if frame.tag != "_":
                indices, tokens = zip(*[(tok.idx - 1, tok.text) for tok in frame.tokens])
                all_frames.append({"predicate_sense": frame.tag, "predicate_word": tokens, "predicate_ix": indices})
    return all_frames


def get_aligned_pred(src_sentence, src_pos, src_preds, pred_tgt):
    def get_sentence_lemmas(src_sentence):
        lemmasent = []
        for w in src_sentence:
            lemmas = lemmatizer(w, u"VERB")
            lemmasent.append(lemmas)
        return lemmasent

    def default_pred():
        sense_lemmas = lemmatizer(pred_tgt[2].split(".")[0], u"VERB")
        sentence_lemmas = get_sentence_lemmas(src_sentence)
        for i, tup in enumerate(sentence_lemmas):
            for wl in tup:
                if wl in sense_lemmas and src_pos[i] == u"VERB":
                    return i, wl, "<UNK>", "V"
        return -1, "-", "<NO-PRED>", "-"

    def choose_pred():
        for sp in src_preds:
            if sp["predicate_sense"] == pred_tgt[2]:
                return sp["predicate_ix"][-1], sp["predicate_word"][-1], sp["predicate_sense"], "V"
        return -1, "-", "<NO-PRED>", "-"

    matched_pred = choose_pred()
    if matched_pred[0] == -1 and len(src_pos) > 0: matched_pred = default_pred()
    src_pred_ix = matched_pred[0]
    IOB_src = ["O" for x in src_sentence]
    if src_pred_ix > 0: IOB_src[src_pred_ix] = "B-V"
    return IOB_src, matched_pred


def get_txt_sentences(filename):
    sents = []
    for line in open(filename).readlines():
        sents.append(line.strip("\n"))
    return sents


def BIO_to_Sequence(word_list, tag_list):
    tagged_tokens = []
    open_tag, close_tag = "", ""
    for word, tag in zip(word_list, tag_list):
        if "B-" in tag:
            if close_tag == "":
                tagged_tokens.append("(#")
                close_tag = tag[2:] + ")"
                tagged_tokens.append(word)
            else:
                tagged_tokens.append(close_tag)
                tagged_tokens.append("(#")
                tagged_tokens.append(word)
                close_tag = tag[2:] + ")"
        elif "I-" in tag:
            tagged_tokens.append(word)
        else:
            if close_tag != "":
                tagged_tokens.append(close_tag)
                close_tag = ""
            tagged_tokens.append(word)
    if close_tag != "": tagged_tokens.append(close_tag)
    return tagged_tokens


def mark_sequence(word_list, tag_list):
    marked_seq = []
    try:
        pred_ix = tag_list.index("B-V")
        for i, word in enumerate(word_list):
            if i == pred_ix: marked_seq.append("<PRED>")
            marked_seq.append(word)
        return marked_seq
    except:
        return word_list


def make_mono_files(file_props, append_in_file=False):
    print("---------------------\nProcessing {} file...".format(file_props["in"]))
    no_preds, no_verb, written_in_file = 0, 0, 0
    sentences = read_conll(file_props["in"], conll_token=file_props["token_type"])
    file_mode = "a" if append_in_file else "w"
    json_file = io.open(file_props["out"], file_mode, encoding='utf8')
    for sent in sentences:
        seq_obj = {}
        my_sent = sent.get_words()
        # print(sent.get_sentence() + "\n")
        # sent.show_pred_args()
        per_predicate = list(sorted(sent.BIO_sequences.items(), key=lambda x: x[0][0]))
        if len(per_predicate) > 0:
            for ix, (pred_sense, seq) in enumerate(per_predicate):
                seq_obj["seq_words"] = my_sent
                seq_obj["BIO"] = seq
                seq_obj["pred_sense"] = sent.predicates[ix]
                if "B-V" in seq:
                    seq_obj["seq_marked"] = mark_sequence(my_sent, seq)
                    seq_obj["seq_tag_tokens"] = BIO_to_Sequence(my_sent, seq)
                else:
                    seq_obj["seq_tag_tokens"] = my_sent
                    no_verb += 1
                    # print("SKEPT!\n{}\n{}\n{}".format(my_sent, seq, sent.show_pred_args()))
                seq_obj["src_lang"] = file_props["lang"]
                seq_obj["tgt_lang"] = "<" + file_props["lang"][1:-1] + "-SRL>"
                json_file.write(json.dumps(seq_obj) + "\n")
                written_in_file += 1
        else:
            no_preds += 1
            generic_bio = ["O" for x in my_sent]
            seq_obj["seq_words"] = my_sent
            seq_obj["BIO"] = generic_bio
            seq_obj["pred_sense"] = (-1, "-", "<NO-PRED>", "-")
            seq_obj["seq_marked"] = my_sent
            seq_obj["seq_tag_tokens"] = my_sent
            seq_obj["src_lang"] = file_props["lang"]
            seq_obj["tgt_lang"] = "<" + file_props["lang"][1:-1] + "-SRL>"
            json_file.write(json.dumps(seq_obj) + "\n")
            written_in_file += 1
    print("IN: {} --> OUT: {}\nFound {} in CoNLL --> Wrote {} in JSON".format(file_props["in"], file_props["out"],
                                                                              len(sentences), written_in_file))
    print("Sentences without predicates = {}\nSkept Malformed Sentences = {}".format(no_preds, no_verb))


def make_parallel_files(file_props, append_in_file=False):
    # Prepare Output JSON File
    file_mode = "a" if append_in_file else "w"
    json_file = io.open(file_props["output"], file_mode, encoding='utf8')
    # Load Source Frame Tagger (Flair)
    if file_props["src_lang"] == "<EN>":
        flair_frame_tagger = SequenceTagger.load('frame')
    else:
        flair_frame_tagger = None
    # Get proper tokenizer
    tokenize_text = file_props.get("tokenize", False)
    if tokenize_text:
        src_spacy = get_lang(file_props["src_lang"])
    else:
        src_spacy = None
    # Load Src and Tgt sentences
    src_sentences = get_txt_sentences(file_props["src_txt"])
    tgt_sentences = read_conll(file_props["tgt_conll"], conll_token=CoNLL09_Token)
    assert len(src_sentences) == len(tgt_sentences), "Src len [{}] and Tgt len [{}] don't match!".format(len(src_sentences), len(tgt_sentences))

    # Construct JSON
    written_in_file = 0
    for tgt_ix, tgt_sent in enumerate(tgt_sentences):
        seq_obj = {}
        per_predicate = list(sorted(tgt_sent.BIO_sequences.items(), key=lambda x: x[0][0]))
        for ix, (pred_sense, tgt_bio) in enumerate(per_predicate):
            src_sentence, src_pos = tokenize_postag_sentence(src_sentences[tgt_ix], src_spacy)

            seq_obj["seq_words"] = src_sentence
            if "B-V" in tgt_bio:
                src_frames = get_source_frames(src_sentence, flair_frame_tagger)
                src_bio, src_pred = get_aligned_pred(src_sentence, src_pos, src_frames, tgt_sent.predicates[ix])
                if not "B-V" in src_bio: continue # Skip sentence for which we didn't find a matching source predicate
                seq_obj["BIO"] = src_bio
                seq_obj["pred_sense_origin"] = src_pred
                seq_obj["pred_sense"] = tgt_sent.predicates[ix]
                seq_obj["seq_marked"] = mark_sequence(src_sentence, src_bio)
                tagged_seq = BIO_to_Sequence(tgt_sent.get_words(), tgt_bio)
                seq_obj["seq_tag_tokens"] = tagged_seq
            else:
                generic_bio = ["O" for x in src_sentence]
                seq_obj["BIO"] = generic_bio
                seq_obj["pred_sense_origin"] = (-1, "-", "<NO-PRED>", "-")
                seq_obj["pred_sense"] = (-1, "-", "<NO-PRED>", "-")
                seq_obj["seq_marked"] = src_sentence
                seq_obj["seq_tag_tokens"] = tgt_sent.get_words()
            seq_obj["src_lang"] = file_props["src_lang"]
            seq_obj["tgt_lang"] = file_props["tgt_lang"]
            json_file.write(json.dumps(seq_obj) + "\n")
            written_in_file += 1
    print(" {} --> {}\n TOTAL: {} --> {}".format(file_props["tgt_conll"], file_props["output"], len(tgt_sentences), written_in_file))


if __name__ == "__main__":
    # Create MONOLINGUAL Span-Based CoNLL05 Datasets in JSON Format
    data_in_path = "datasets/raw/"
    data_out_path = "datasets/json/"

    all_files = [{"in":  data_in_path + "CoNLL2005-trial.txt",
                  "out":  data_out_path + "EN_conll05_trial.json",
                  "lang": "<EN>",
                  "token_type": CoNLL05_Token},
                 {"in": data_in_path + "CoNLL2005-test-trial.txt",
                  "out": data_out_path + "EN_conll05_test_trial.json",
                  "lang": "<EN>",
                  "token_type": CoNLL05_Test_Token}
                 ]

    for file in all_files:
        make_mono_files(file)

    # Create MONOLINGUAL Dependency-Based CoNLL09 Datasets in JSON Format
    all_files = [{"in":  data_in_path + "CoNLL2009-ST-English-trial.txt",
                  "out":  data_out_path + "EN_conll09_trial.json",
                  "lang": "<EN>",
                  "token_type": CoNLL09_Token},
                 {"in": data_in_path + "CoNLL2009-ST-German-trial.txt",
                  "out": data_out_path + "DE_conll09_trial.json",
                  "lang": "<DE>",
                  "token_type": CoNLL09_Token},
                 {"in": data_in_path + "CoNLL-vdPlas-French-trial.txt",
                  "out": data_out_path + "FR_conll09_trial.json",
                  "lang": "<FR>",
                  "token_type": CoNLL09_Token},
                 {"in": data_in_path + "CoNLL-vdPlas-French-GOLD.txt",
                  "out": data_out_path + "FR_conll09_GOLD.json",
                  "lang": "<FR>",
                  "token_type": CoNLL09_FrenchTestToken}
                 ]

    for file in all_files:
        make_mono_files(file)

    # Create CROSS-LINGUAL Dependency-Based (Akbik and van der Plas datasets)
    lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
    all_files = [{"src_txt": data_in_path + "CrossLang_ENDE_EN_trial.txt",
                  "tgt_conll": data_in_path + "CrossLang_ENDE_DE_trial.conll09",
                  "output": data_out_path + "En2DeSRL.json",
                  "src_lang": "<EN>",
                  "tgt_lang": "<DE-SRL>",
                  "tokenize": True}
                ]

    for file in all_files:
        make_parallel_files(file)
