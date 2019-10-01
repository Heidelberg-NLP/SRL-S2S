import json, argparse, random
from functools import reduce


def linear_gluer(filehandle, generators):
    for gen in generators:
        for x in gen:
            filehandle.write(json.dumps(x)+"\n")


def uniform_sampler_gluer(filehandle, generators):
    keep_going = [True for x in generators]
    counter = 0
    exceptions = []
    while reduce((lambda x, y: x or y), keep_going):
        ix = random.randint(0, len(generators) - 1)
        if ix in exceptions: continue
        gen = generators[ix]
        counter += 1
        try:
            filehandle.write(json.dumps(next(gen))+"\n")
        except StopIteration:
            keep_going[ix] = False
            exceptions.append(ix)


def get_json_lines(filename):
    with open(filename) as f:
        for line in f:
            yield json.loads(line)

if __name__ == "__main__":
    """
        RUN EXAMPLE: 
            python pre_processing/json_gluer.py  -p datasets/conll09/multi_file_data -d En2De.mini.json En2Fr.mini.json -o all.mini.json
            
            ----- EXAMPLE: Multilingual 1-to-1 files (EN + DE) -----
            python pre_processing/json_gluer.py  -p datasets/json/ \
            -d EnglishJSON/conll09_train.json FrenchJSON/conll09_train.json GermanJSON/conll09_train.json \
            -o one2one_multilingual.ALL.json
        """
    # Read arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Filepath where the inputs are and the where output file will be saved', required=True)
    parser.add_argument('-d', '--datasets', help='JSON Files that will be joined', required=True, nargs='+')
    parser.add_argument('-o', '--output', help='Output JSON with all the records from input data', required=True)
    args = parser.parse_args()

    # random.seed(1989)

    data_generators = [get_json_lines(args.path + "/" + x) for x in args.datasets]

    output_file = open(args.path+"/"+args.output, "w")

    # linear_gluer(output_file, data_generators)
    uniform_sampler_gluer(output_file, data_generators)
