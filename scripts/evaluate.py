import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import logging
import argparse
from evaluator import CommonEvaluator, F1Evaluator

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

TOKENIZER_PATH = "ckpts/scibert_scivocab_uncased"

TASKS_EVALUATOR = {
    "trans_iupac": ["common"],
    "general_desp": ["common"],
    "struct_cap": ["f1"]
}

def evaluate(exp_name="mol-vl_7b"):

    logging.info(f"Start Evaluation of {exp_name}.")

    with open(os.path.join("outputs/inference_logs", exp_name, "logs.json"), 'r') as f:
        logs = json.load(f)

    res = dict()
    for task in TASKS_EVALUATOR:
        try:
            log = logs[task]
            res[task] = dict()

            logging.info(f"[Task] Evaluating {task}...")

            for i in TASKS_EVALUATOR[task]:
                if i == "common":
                    evaluator = CommonEvaluator(TOKENIZER_PATH)
                    res[task]["common"] = evaluator.evaluate(log)
                
                elif i == "f1":
                    evaluator = F1Evaluator()

                    gts = list()
                    preds = list()
                    for j in log:
                        gts.append(j["gt"][:-1].replace("The functional group ", "").replace(" is highlighted", "").replace("This molecule contains several functional groups, including ", "").replace(" and", "").replace("This molecule contains the ", "").replace(" functional group", "").split(", "))
                        preds.append(j["pred"][:-1].replace("The functional group ", "").replace(" is highlighted", "").replace("This molecule contains several functional groups, including ", "").replace(" and", "").replace("This molecule contains the ", "").replace(" functional group", "").split(", "))
                    res[task]["f1"] = evaluator.evaluate(gts, preds)
        except:
            raise ValueError(f"[Task] {task} is not supported.")

    with open(os.path.join("outputs/inference_logs", exp_name, "results.json"), 'w') as f:
        json.dump(res, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="mol-vl_7b")

    args = parser.parse_args()

    evaluate(exp_name=args.exp_name)
