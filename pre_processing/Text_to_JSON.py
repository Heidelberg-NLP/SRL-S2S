"""
    RUN EXAMPLE:
        python pre_processing/Text_to_JSON.py --path datasets/raw/ \
            -s mini_europarl-v7.de-en.en -t mini_europarl-v7.de-en.de -o datasets/json/MiniEuroparl.en_to_de.json \
            --src_key "<EN>" --tgt_key "<DE>"
            
        python pre_processing/Text_to_JSON.py --source_file datasets/raw/mini_europarl-v7.de-en.en \
             --output datasets/json/MiniEuroparl.PREDICT.json \
             --src_key "<EN>" --tgt_key "<DE-SRL>" \
             --predict_frames True \
             --sense_dict datasets/aux/En_De_TopSenses.tsv
"""

import json, spacy, argparse
from flair.data import Sentence
from flair.models import SequenceTagger
included, skept = 0, 0


def get_tokenizer(lang):
    if lang =="<EN>":
        return spacy.load("en")
    elif lang == "<DE>":
        return spacy.load("de")
    elif lang == "<FR>":
        return spacy.load("fr")
    else:
        return None


def tokenize_sentence(sentence_str, spacy_lang=None):
    if not spacy_lang:
        return sentence_str.split()
    else:
        return [tok.text for tok in spacy_lang.tokenizer(sentence_str)]


def get_most_frequent_senses(filename, threshold):
    sense_dict = {}
    with open(filename) as f:
        for i, line in enumerate(f):
            sense, count = line.split("\t")
            count = int(count.strip("\n"))
            if count >= threshold:
                sense_dict[sense] = int(count)
    return sense_dict


def get_frames(sentence_tokens, frame_tagger, valid_senses, sent_index):
    if not frame_tagger: return []
    all_frames = []
    sent_included, sent_total = 0, 0
    global skept
    global included
    sentence_obj = Sentence(" ".join(sentence_tokens))
    frame_tagger.predict(sentence_obj)
    for frame in sentence_obj.get_spans('frame'):
        indices, tokens = zip(*[(tok.idx - 1, tok.text) for tok in frame.tokens])
        if frame.tag != "_":
            sent_total += 1
            if valid_senses:
                if frame.tag in valid_senses:
                    all_frames.append({"predicate_sense": frame.tag, "predicate_word": tokens, "predicate_ix": indices})
                    sent_included += 1
                else:
                    skept += 1
            else:
                all_frames.append({"predicate_sense": frame.tag, "predicate_word": tokens, "predicate_ix": indices})
                sent_included += 1

    included += sent_included
    print("Sentence {} has {} senses in total".format(sent_index, sent_total))
    if sent_total> 0: print("Only kept {} [senses ABOVE the threshold]".format(sent_included))
    print("Total Found so far: {}".format(included))
    print("---------")
    return all_frames


def sentences2JSON(src_filename, output, src_key, tgt_key, predict_frames):
    counter = 0
    json_output = open(output, "w")
    src_tokenizer = get_tokenizer(src_key)
    with open(src_filename) as src_file:
        for s in src_file:
            src = s.strip("\n")
            if len(src) == 0: continue
            counter += 1
            source_tokens = tokenize_sentence(src, src_tokenizer)
            obj = {"seq_words": source_tokens,
                   "seq_tag_tokens": None,
                   "BIO": ["O" for x in source_tokens],
                   "src_lang": src_key,
                   "tgt_lang": None
                   }
            if predict_frames:
                sent_frames = get_frames(source_tokens, flair_frame_tagger, valid_senses, counter)
                for frame in sent_frames:
                    tmp = ["O" for x in source_tokens]
                    tmp[frame["predicate_ix"][-1]] = "B-V"
                    obj["BIO"] = tmp
                    obj["tgt_lang"] = tgt_key
                    obj["pred_sense_origin"] = frame
                    json_output.write(json.dumps(obj) + "\n")
            else:
                json_output.write(json.dumps(obj) + "\n")
    return None


def file_pair2JSON(src_filename, tgt_filename, output, src_key, tgt_key, predict_frames):
    json_output = open(output, "w")
    src_tokenizer = get_tokenizer(src_key)
    tgt_tokenizer = get_tokenizer(tgt_key)
    counter = 0
    with open(src_filename) as src_file, open(tgt_filename) as tgt_file:
        for s, t in zip(src_file, tgt_file):
            counter += 1
            src = s.strip("\n")
            tgt = t.strip("\n")
            if len(src) == 0 or len(tgt) == 0: continue
            source_tokens = tokenize_sentence(src, src_tokenizer)
            target_tokens = tokenize_sentence(tgt, tgt_tokenizer)
            obj = {"seq_words": source_tokens,
                   "seq_tag_tokens": target_tokens,
                   "BIO": ["O" for x in source_tokens],
                   "src_lang": src_key,
                   "tgt_lang": tgt_key
                    }
            if predict_frames:
                sent_frames = get_frames(source_tokens, flair_frame_tagger, valid_senses, counter)
                for frame in sent_frames:
                    tmp = ["O" for x in source_tokens]
                    tmp[frame["predicate_ix"][-1]] = "B-V"
                    obj["BIO"] = tmp
                    obj["tgt_lang"] = tgt_key
                    obj["pred_sense_origin"] = frame
                    json_output.write(json.dumps(obj) + "\n")
            else:
                json_output.write(json.dumps(obj)+"\n")


if __name__ == "__main__":
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Filepath where the input is and the where output file will be saved', default="")
    parser.add_argument('-s', '--source_file', help='JSON File with the model outputs', required=True)
    parser.add_argument('-t', '--target_file', help='If files are separated per language', default=None)

    parser.add_argument('-ks', '--src_key', help='key for the JSON, compatible with allennlp DatasetReader!', required=True)
    parser.add_argument('-kt', '--tgt_key', help='OPTIONAL: key for the JSON, compatible with allennlp DatasetReader!', required=True)
    parser.add_argument('-o',  '--output', help='JSON file where the output will be saved', required=True)

    parser.add_argument('-sd', '--sense_dict', help='TSV file with the dataset senses', default=None)
    parser.add_argument('-th', '--threshold', help='Keep only predicates that have this minimum count in the sense dict', default=50)
    parser.add_argument('-fr', '--predict_frames', help='True to enable frame prediction of sentences', default='False')
    args = parser.parse_args()

    valid_senses = None
    flair_frame_tagger = None

    if args.predict_frames == "True":
        flair_frame_tagger = SequenceTagger.load('frame')
        if args.sense_dict:
            valid_senses = get_most_frequent_senses(args.sense_dict, threshold=args.threshold)

    if args.target_file:
        file_pair2JSON(src_filename=args.path + args.source_file,
                       tgt_filename=args.path + args.target_file,
                       output=args.output,
                       src_key=args.src_key,
                       tgt_key=args.tgt_key,
                       predict_frames=args.predict_frames)
    else:
        sentences2JSON(src_filename=args.path + args.source_file,
                       output=args.output,
                       src_key=args.src_key,
                       tgt_key=args.tgt_key,
                       predict_frames=args.predict_frames)

    print("DONE! A JSON dataset file was created in {}".format(args.output))

