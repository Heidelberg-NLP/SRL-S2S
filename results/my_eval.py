import sys, os
import tempfile

import json
import numpy as np
import argparse

from log.set_log import set_log
logger = set_log("eval.log")

from allennlp.models.semantic_role_labeler import convert_bio_tags_to_conll_format
DEBUG_MATCH = True
path_eval_script = "result/srl-eval.pl"

class PredJsonReader(AllenNlpTestCase):
    ''' predictor 実行後に出力される output-file.js を読み込んで srl-eval.pl 実行用の prop ファイルを作成 '''
    def __init__(self,
                 saved_dir: str,
                 json_file_from_predictor: str,
                 test: str = 'wsj',
                 oracle: str = 'min',
                 ) -> None:
        
        self._saved_dir = saved_dir if not saved_dir.endswith('/') else saved_dir[:-1]
        self._predictor_output = os.path.abspath(os.path.join(self._saved_dir, json_file_from_predictor))
        
        self._gold_prop = os.path.abspath('datasets/conll05/conll05.test.{}.prop'.format(test))
        self._pred_prop = os.path.abspath(os.path.join(self._saved_dir, 'eval.test.{}.{}.prop'.format(test, oracle)))
        
        sys.path.append(self._predictor_output)
        sys.path.append(self._gold_prop)
        sys.path.append(self._pred_prop)
        

    ### instance 毎に yield ###
    def _reader(self):
        """
        line_obj:
            * metadata:             Dict
            * predicted_log_probs:  List[ float ]           #beam
            * predictions:          List[ List(ids) ]       #beam
            * predicted_tokens:     List[ List(tokens) ]    #beam
        ---------------------------
        ## metadata
            * souruce_tokens:               ['<EN-SRL>', 'Some', ..., 'i.'] 
            * verb:                         'installed'
            * src_lang:                     '<EN>'
            * tgt_lang:                     '<EN-SRL>'
            * original_BIO:                 ['B-A1', 'I-A1', ..., 'O']
            * original_predicate_senses:    []
            * predicate_senses:             [5, 'installed', '-', 'VBN']
            * original_target:              ['(#', 'Some', ..., '.']
        """
        count_different_seq_len, count_inappropriate_bracket, count_without_pred = 0, 0, 0
        
        for instance_no, line in enumerate(open(self._predictor_output), start=1):
            line_obj = json.loads(line.strip())
            metadata = line_obj['metadata']
            
            sentence = metadata['source_tokens'][1:]        # '<EN-SRL>' を除く
            verb = tuple(metadata['predicate_senses'][:2])  # (v_idx, verb)
            
            predicted_target = line_obj['predicted_tokens'][0] if not DEBUG_MATCH else \
                               metadata['original_target']
            
            bio_gold = metadata['original_BIO']
            bio_pred, invalid_bracket = self.create_predicted_BIO(predicted_target, metadata)
            
            conll_formatted_gold_tag = convert_bio_tags_to_conll_format(bio_gold)
            conll_formatted_predicted_tag = convert_bio_tags_to_conll_format(bio_pred)
            
            ### counter ###
            if len(bio_gold) != len(bio_pred):      # pred と gold の系列長が異なる
                count_different_seq_len += 1
            if invalid_bracket:                     # 括弧づけが不適切
                count_inappropriate_bracket += 1
            if verb[0] == -1:                       # target 述語が存在しない
                count_without_pred += 1
            
            yield verb, sentence, conll_formatted_predicted_tag, conll_formatted_gold_tag, line_obj
        
        logger.info("COUNT:")
        logger.info(" - instances: %d" % instance_no)
        logger.info(" - different_seq_len: %d" % count_different_seq_len)
        logger.info(" - inappropriate_bracket: %d" % count_inappropriate_bracket)
        logger.info(" - without_pred: %d" % count_without_pred)

    
    def create_predicted_BIO(self, predicted_target: list, metadata) -> list:
        props, out = [], []
        prop = 'O'
        prev = None
        invalid_bracket = False
        close_bracket = 0
        
        # predicted_BIO = reversed(props)
        for word in reversed(predicted_target):
            if word.endswith(')') and len(word) > 1:
                if close_bracket == 1: 
                    invalid_bracket = True
                    continue   # 削除
                close_bracket += 1
                prop = word[:-1]
            elif word == '(#':
                if close_bracket == 0: 
                    invalid_bracket = True
                    continue   # 削除
                if not prop == "O":
                    props[-1] = "B-" + prop
                prop = 'O'
                close_bracket -= 1
            else:
                if prop == "O":
                    props.append(prop)
                else:
                    props.append("I-" + prop)
        
        out = [p for p in reversed(props)]        
        return out, invalid_bracket


    def check_bracket(self, sent:list) -> bool:
        bracket = 0
        for token in sent:
            if bracket < 0 or 1 < bracket: 
                return False
            bracket = bracket+1 if token.startswith('(#') else \
                      bracket-1 if token.endswith(')') and len(token) > 1 else \
                      bracket
        return True


    ### pred_prop に conll形式 で書き込み ###
    def create_conll_format(self, oracle):
        prev_sent = ''
        buffer_sent = []
        buffer_obj = []
        with open(self._pred_prop, 'w') as fo, open(self._gold_prop, 'r') as fr:
            gold_lines = self.read_lines_for_prop(fr)
            gold_verbs = [e[0] for e in gold_lines if e != '-']
            isValid, count_invalid = True, 0
            
            for idx, (verb, sent, pred, gold, line_obj) in enumerate(self._reader()):
                if not len(sent) == len(pred):  # pred と gold の系列長が異なる
                    isValid = False
                    pred = ['*' for _ in gold]  # 一列目を'-' 埋めしたい
                
                ### 新しい文の場合
                if not prev_sent == sent:
                    if buffer_sent:     ## 初期値の回避
                        # if isValid is False: count_invalid += 1
                        self.write_buffer_sent(fo, buffer_sent, gold_lines, isValid, oracle, buffer_obj)
                        buffer_sent = []    
                        buffer_obj = []
                        gold_lines = self.read_lines_for_prop(fr)
                        gold_verbs = [e[0] for e in gold_lines if e != '-']
                    
                    isValid = True
                    # 新しい文が来るまで buffer に溜め込む
                    buffer_obj.append(line_obj)
                    if verb[0] == -1: # 述語が存在しない -> target verb 列を - で埋める
                        buffer_sent = [['-'] for _ in range(len(sent))]
                    else:
                        buffer_sent = [[verb[1], pred[i]] if i == verb[0] else ['-', pred[i]] for i in range(len(sent))]
                    
                ### 同一文の場合（二列目以降）
                else:
                    #assert len(pred) == len(buffer_sent), '同一文だが buffer と長さが異なる'
                    buffer_obj.append(line_obj)
                    for word_idx, index_word in enumerate(buffer_sent):
                        index_word.append(pred[word_idx])   # 列追加
                        if word_idx == verb[0]:
                            index_word[0] = verb[1]         # target verb 列を V-B に書き換え
            
                prev_sent = sent
            
            ## 末文
            self.write_buffer_sent(fo, buffer_sent, gold_lines, isValid, oracle, buffer_obj)
            
        logger.info('write to ... %s' % self._pred_prop)

    
    def read_lines_for_prop(self, prop_file) -> list:
        gold_lines = []
        gold_line = prop_file.readline()
        while (len(gold_line.strip()) > 0):
            gold_lines.append(gold_line.split())
            gold_line = prop_file.readline()
        return gold_lines
    
    
    def write_buffer_sent(self, fo, buffer_sent, gold_lines, isValid, oracle, bf_obj):
        if isValid:
            for word_idx, (line_word, gold) in enumerate(zip(buffer_sent, gold_lines)):
                output = []
                if gold[1:] != line_word[1:]:
                    verbs = [bf["metadata"]["verb"] for bf in bf_obj]
                    bio_gold = [bf["metadata"]["original_BIO"][word_idx] for bf in bf_obj]
                    pred_sents = [bf["predicted_tokens"][0] for bf in bf_obj]

                    print(word_idx, bf_obj[0]["metadata"]["source_tokens"][word_idx+1], bio_gold, verbs)
                    print("gold", gold)
                    print("pred", line_word)
                    for sent in pred_sents:
                        print(sent)
                    #if not any(["C-V" in e for e in gold]):
                    #    import ipdb; ipdb.set_trace()
                output.append(gold[0])
                for col, (b, g) in enumerate(zip(line_word[1:], gold[1:])):     # 二列目以降
                    if "C-V" in g:
                        output.append(g)
                        #import ipdb; ipdb.set_trace()
                    else:
                        output.append(b)
                fo.write(self.write_output(output))
            fo.write('\n')
        else:
            for word_idx, (line_word, line_gold) in enumerate(zip(buffer_sent, gold_lines)):
                output = [gold_lines[word_idx][0]]
                for col, (_, g) in enumerate(zip(line_word[1:], line_gold[1:])):  # 二列目以降
                    if oracle == 'min':
                        output.append('*')
                    elif oracle == 'max':
                        output.append(g)
                fo.write(self.write_output(output))
            fo.write('\n')

    
    def write_output(self, out:list) -> str:
        """
        gold.prop と間隔を揃える
        """
        exception = ["R-AM-TMP", "R-AM-ADV", "R-AM-EXT", "R-AM-CAU", "R-AM-MNR", "R-AM-LOC"]
        line = out[0]
        right_overflow = len(out[0]) - 1
        if len(out) > 1:
            for i, token in enumerate(out[1:]):
                prev = out[i]
                space = 23 if i == 0 else 15
                asta = token.rfind('*')
                n = space - asta - right_overflow
                if any(map(lambda x: x in token, exception)): #例外
                    n += 1
                if any(map(lambda x: x in prev, exception)):
                    n -= 1
                right_overflow = len(token) - asta - 1
                line += " "*n + token
            n = 6 - right_overflow
            #if any(map(lambda x: any([x in o for o in out]), exception)):
            #    n -= 1
            if any(map(lambda x: x in out[-1], exception)):
                n -= 1
            line += " "*n + "\n"
        else:
            line += " "*14 + "\n"
        return line


    ### predictor 実行後に出力される output-file.js を読み込んで，allennlp の SrlEvalScorer を __call__ ###
    def srl_eval(self) -> None:
        
        batch_verb_indices, batch_sentences, batch_conll_formatted_predicted_tags, batch_conll_formatted_gold_tags = self.get_batch_instances()
        
        self._srl_scorer(
                    batch_verb_indices,
                    batch_sentences,
                    batch_conll_formatted_predicted_tags,
                    batch_conll_formatted_gold_tags
                    )
        
        metrics = self._srl_scorer.get_metric()
        self.write_metrics(metrics)


    def get_batch_insntaces(self):
        verb_indices, sentences, conll_predicts, conll_golds = [], [], [], []
        for verb, sent, pred, gold in self._reader():
            verb_indices.append(verb[0])
            sentences.append(sent)
            conll_predicts.append(pred)
            conll_golds.append(gold)
        return verb_indices, sentences, conll_predicts, conll_golds


    def write_metrics(self, metrics):
        test_target = ['A0', 'A1', 'A2', 'AM-TMP', 'AM-LOC', 'overall']
        #test_target = ['overall']
        sys.stdout.write('\n\tPREC\tREC\tF1\n')
        
        for t in test_target:
            # assert_allclose(metrics['precision-' +t], 1.0)
            # assert_allclose(metrics['recall-' +t], 1.0)
            # assert_allclose(metrics['f1-measure-' +t], 1.0)
            
            sys.stdout.write(
                    '{}\t{:.3f}\t{:.3f}\t{:.3f}\n'.format(
                        t,
                        metrics['precision-' +t],
                        metrics['recall-' +t],
                        metrics['f1-measure-' +t]
                        )
                    )



def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('saved_models', type=str)
    parser.add_argument('--test', default='wsj', type=str, help='wsj/brown')
    parser.add_argument('--oracle', default='min', type=str, help='min/max')
    parser.set_defaults(no_thres=False)
    return parser


if __name__ == '__main__':

    parser = create_arg_parser()
    args = parser.parse_args()
    
    predicted_json = 'predicted.from.test.{}.json'.format(args.test)    # from predictor
    predicted_json = os.path.join('saved_models', predicted_json)
    
    logger.debug("read from ... %s" % predicted_json)
    logger.debug("test ... %s" % args.test)
    logger.debug("oracle ... %s" % args.oracle)

    creator = PredJsonReader(
            args.saved_dir, # saved_dir
            predicted_json, # json file from predictor
            args.test,      # wsj/brown
            args.oracle,    # min/max
            )

    creator.create_conll_format(args.oracle)
    sys.stdout.write('DONE!\n\n')
    sys.stdout.write('$ perl {} {} {}\n'.format(
            path_eval_script, 
            creator._gold_prop,
            creator._pred_prop
            )
        )

