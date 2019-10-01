from overrides import overrides
from allennlp.common.util import JsonDict, sanitize, group_by_count
from allennlp.data import DatasetReader, Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('seq2seq-srl')
class Seq2SeqPredictor(Predictor):
    """Predictor wrapper for the CopyNetSeq2Seq for SRL"""
    def predict(self, json_dict: JsonDict) -> JsonDict:
        return self.predict_json({"source": " ".join(json_dict["seq_marked"])})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        source_key = "seq_words" # "seq_marked"
        return self._dataset_reader.text_to_instance(source_key, line_obj=json_dict)
