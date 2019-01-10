from typing import Dict, List, Any

from overrides import overrides
import torch
import numpy as np
from allennlp.models.ensemble import Ensemble
from allennlp.training.metrics import SquadEmAndF1
import bidaf_utils as bidut
from torch.nn.functional import nll_loss
from bidaf_model import BidirectionalAttentionFlow_1
#@Model.register("QA-ensemble")
class QA_ensemble(Ensemble):

    """
    Ensemble of models for Q&A. 
    Initially meant for BiDAF models but extendable to other models
    and dataset provided the same interface
    """

    def __init__(self, submodels: List[object], load_models = False) -> None:
        """
        TODO: Make the output the same as for the BiDAF, not a simplification
        If load_models = True the algorithm will call the load() function of the objects
        which should load them from disk to the RAM. 
        """
        super().__init__(submodels)
        self.cf_a = submodels[0].cf_a
        self._squad_metrics = SquadEmAndF1()


    @overrides
    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None,
                get_sample_level_information = True) -> Dict[str, torch.Tensor]:
        """
        WE LOAD THE MODELS ONE INTO GPU  ONE AT A TIME !!!
        """
        
        subresults = []
        for submodel in self.submodels:
            submodel.to(device = submodel.cf_a.device)
            subres = submodel(question, passage, span_start, span_end, metadata, get_sample_level_information)
            submodel.to(device = torch.device("cpu"))
            subresults.append(subres)

        batch_size = len(subresults[0]["best_span"])

        best_span = merge_span_probs(subresults)
        output = {
                "best_span": best_span,
                "best_span_str": [],
                "models_output": subresults
        }
        if (get_sample_level_information):
            output["em_samples"] = []
            output["f1_samples"] = []
                
        for index in range(batch_size):
            if metadata is not None:
                passage_str = metadata[index]['original_passage']
                offsets = metadata[index]['token_offsets']
                predicted_span = tuple(best_span[index].detach().cpu().numpy())
                start_offset = offsets[predicted_span[0]][0]
                end_offset = offsets[predicted_span[1]][1]
                best_span_string = passage_str[start_offset:end_offset]
                output["best_span_str"].append(best_span_string)

                answer_texts = metadata[index].get('answer_texts', [])
                if answer_texts:
                    self._squad_metrics(best_span_string, answer_texts)
                    if (get_sample_level_information):
                        em_sample, f1_sample = bidut.get_em_f1_metrics(best_span_string,answer_texts)
                        output["em_samples"].append(em_sample)
                        output["f1_samples"].append(f1_sample)
        if (get_sample_level_information):
            # Add information about the individual samples for future analysis
            output["span_start_sample_loss"] = []
            output["span_end_sample_loss"] = []
            for i in range (batch_size):
                
                span_start_probs = sum(subresult['span_start_probs'] for subresult in subresults) / len(subresults)
                span_end_probs = sum(subresult['span_end_probs'] for subresult in subresults) / len(subresults)
                span_start_loss = nll_loss(span_start_probs[[i],:], span_start.squeeze(-1)[[i]])
                span_end_loss = nll_loss(span_end_probs[[i],:], span_end.squeeze(-1)[[i]])
                
                output["span_start_sample_loss"].append(float(span_start_loss.detach().cpu().numpy()))
                output["span_end_sample_loss"].append(float(span_end_loss.detach().cpu().numpy()))
        return output
    def set_posterior_mean(self, value = True):
        for submodel in self.submodels:
            submodel.set_posterior_mean(value)
            
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._squad_metrics.get_metric(reset)
        return {
                'em': exact_match,
                'f1': f1_score,
        }

