import argparse, json


def read_text_file(filename):
    lines = []
    with open(filename) as f:
        for line in f:
            lines.append(line.split())
    return lines


def read_srl_json_output(filename, is_generator=False):
    instances = []
    with open(filename) as f:
        for line in f:
            obj = json.loads(line)
            if is_generator:
                yield obj
            else:
                instances.append(obj)
    return instances


if __name__ == "__main__":
    """
    RUN EXAMPLE:          
        python pre_processing/include_backtranslation.py -j results/srl-nmt-multilang-crosslang-bert-alternate_output_dev.json\
         -b datasets/back_alternate_srl_predictions.txt -l "<DE-SRL>"
    """
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json_original_file', help='JSON File with the model outputs', required=True)
    parser.add_argument('-b', '--backtranslations', help='Text file with the BackTranslations', required=True)
    parser.add_argument('-l', '--lang', help='Target Language whose backtranlsations belong to', required=True)
    args = parser.parse_args()

    count = 0
    output_file = open(args.json_original_file[:-5] + "_back.json", "w")
    back_trans = read_text_file(args.backtranslations)
    print("Backtranslation File Length {}".format(len(back_trans)))

    for obj in read_srl_json_output(args.json_original_file, is_generator=True):
        if obj["metadata"]["tgt_lang"] == args.lang:
            obj["metadata"]["backtranslation"] = back_trans[count]
            count += 1
        output_file.write(json.dumps(obj)+"\n")

    print("Added {} back-translations to the outputs".format(count))
