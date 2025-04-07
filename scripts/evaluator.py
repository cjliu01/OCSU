# conda biomedgpt
import multiprocessing
import logging

import numpy as np
from tqdm import tqdm

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import BertTokenizerFast

from sklearn import metrics

import rdkit
from rdkit import Chem, DataStructs
rdkit.RDLogger.DisableLog('rdApp.*')

class RocAucEvaluator():

    def __init__(self, return_fpr=False, return_tpr=False):

        self.return_fpr = return_fpr
        self.return_tpr = return_tpr

    def evaluate(self, gts, preds):
        """
        :param gts: list(list(float))
        :param preds: list(list(float))
        """
        logging.info("[RocAucEvaluator] Evaluating...")

        fpr, tpr, _ = metrics.roc_curve(gts, preds)
        roc_auc = metrics.auc(fpr, tpr)

        res = dict(roc_auc=roc_auc)
        if self.return_fpr:
            res["fpr"] = fpr
        if self.return_tpr:
            res["tpr"] = tpr
        
        return res


class F1Evaluator():

    def __init__(self, return_recall=True, return_precision=True):

        self.return_recall = return_recall
        self.return_precision = return_precision
    
    def evaluate(self, gts, preds):
        """
        :param gts: list(list(str))
        :param preds: list(list(str))
        """
        logging.info("[F1Evaluator] Evaluating...")

        precision = list()
        recall = list()
        f1 = list()

        for idx in tqdm(range(len(gts))):
            gt = set(gts[idx])
            pred = set(preds[idx])

            inter = gt.intersection(pred)

            p = len(inter)/len(gt)
            r = len(inter)/len(pred) if len(pred) else 0.0
            f = 2*p*r/(p+r) if p+r else 0.0

            precision.append(p)
            recall.append(r)
            f1.append(f)

        res = dict(f1=np.mean(np.array(f1)))
        if self.return_precision:
            res["precision"] = np.mean(np.array(precision))
        if self.return_recall:
            res["recall"] = np.mean(np.array(recall))
        
        return res



class CommonEvaluator():

    def __init__(self, tokenizer_path):

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        logging.info("[CommonEvaluator] Tokenizer Loaded.")

        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    def _tokenize(self, string, max_length=512):

        tokens = self.tokenizer.tokenize(string, truncation=True, max_length=max_length, padding='max_length')
        tokens = list(filter(('[PAD]').__ne__, tokens))
        tokens = list(filter(('[CLS]').__ne__, tokens))
        tokens = list(filter(('[SEP]').__ne__, tokens))

        return tokens


    def evaluate(self, results):
        """
        
        :param results: list(dict(gt, pred))
        """

        probs = {
            "overall": {
                "gt_tokens": [],
                "pred_tokens": [],
                "meteor_score": [],
                "rouge_score": [],
            }
        }

        logging.info("[CommonEvaluator] Evaluating...")

        for i in tqdm(results):

            gt_tokens = self._tokenize(i["gt"])
            pred_tokens = self._tokenize(i['pred'])

            probs["overall"]["pred_tokens"].append(pred_tokens)
            probs["overall"]["gt_tokens"].append([gt_tokens])
            probs["overall"]["meteor_score"].append(meteor_score([gt_tokens], pred_tokens))
            probs["overall"]["rouge_score"].append(self.scorer.score(i['pred'], i["gt"]))
        
        result = {
            "Num Samples:": len(probs["overall"]["gt_tokens"]),
            "BLEU-2" : corpus_bleu(probs["overall"]["gt_tokens"], probs["overall"]["pred_tokens"], weights=(0.5, 0.5)),
            "BLEU-4" : corpus_bleu(probs["overall"]["gt_tokens"], probs["overall"]["pred_tokens"], weights=(0.25, 0.25, 0.25, 0.25)),
            "ROUGE-1": np.mean([rs["rouge1"].fmeasure for rs in probs["overall"]["rouge_score"]]),
            "ROUGE-2": np.mean([rs["rouge2"].fmeasure for rs in probs["overall"]["rouge_score"]]),
            "ROUGE-L": np.mean([rs["rougeL"].fmeasure for rs in probs["overall"]["rouge_score"]]),
            "METEOR" : np.mean(probs["overall"]["meteor_score"])
        }

        return result

def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    from SmilesPE.pretokenizer import atomwise_tokenizer

    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success

def convert_smiles_to_canonsmiles(
        smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)

def tanimoto_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
        tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
        return tanimoto
    except:
        return 0

def compute_tanimoto_similarities(gold_smiles, pred_smiles, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        similarities = p.starmap(tanimoto_similarity, [(gs, ps) for gs, ps in zip(gold_smiles, pred_smiles)])
    
    return similarities

class SmilesEvaluator(object):
    def __init__(self, num_workers=16, tanimoto=True):
        self.num_workers = num_workers
        self.tanimoto = tanimoto

    def _set_goldsmiles(self, gold_smiles):
        self.gold_smiles = gold_smiles
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                     ignore_cistrans=True,
                                                                     num_workers=self.num_workers)
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(gold_smiles,
                                                                   ignore_chiral=True, ignore_cistrans=True,
                                                                   num_workers=self.num_workers)
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)

    def _replace_empty(self, smiles_list):
        """Replace empty SMILES in the gold, otherwise it will be considered correct if both pred and gold is empty."""
        return [smiles if smiles is not None and type(smiles) is str and smiles != "" else "<empty>"
                for smiles in smiles_list]

    def evaluate(self, gt_smiles, pred_smiles, include_details=False):
        logging.info("[SmilesEvaluator] Evaluating...")
        self._set_goldsmiles(gt_smiles)

        results = {}
        if self.tanimoto:
            results['tanimoto'] = np.mean(compute_tanimoto_similarities(self.gold_smiles, pred_smiles))
        # Ignore double bond cis/trans
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                                ignore_cistrans=True,
                                                                num_workers=self.num_workers)
        results['canon_smiles'] = np.mean(np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        if include_details:
            results['canon_smiles_details'] = (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans))
        # Ignore chirality (Graph exact match)
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(pred_smiles,
                                                              ignore_chiral=True, ignore_cistrans=True,
                                                              num_workers=self.num_workers)
        results['graph'] = np.mean(np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral))
        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if '@' in g])
        results['chiral'] = np.mean(chiral[:, 0] == chiral[:, 1]) if len(chiral) > 0 else -1
        return results