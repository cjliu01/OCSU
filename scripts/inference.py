import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
from tqdm import tqdm
import argparse

from utils import Agent

def inference(api_url="http://localhost:8000/v1", \
              exp_name="mol-vl_7b", \
              tasks=["trans_iupac", "trans_smiles", "general_desp", "struct_cap"]):
    
    agent = Agent(api_url)

    with open("data/Vis-CheBI20/test.json", 'r') as f:
        infos = json.load(f)

    log_path = os.path.join("outputs/inference_logs", exp_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    log = dict()
    for task in tasks:
        log[task] = list()

    cnt = 0 
    for info in tqdm(infos):

        task = info['task_name']
        if task not in tasks:
            continue

        img_path = info["images"][0]
        query = info['messages'][0]['content']
        gt = info['messages'][1]['content']

        pred = agent.chat(img_path, query)

        log[task].append(dict(gt=gt, pred=pred))

        cnt += 1
        if cnt % 10 == 0:
            with open(os.path.join(log_path, "logs.json"), 'w') as f:
                json.dump(log, f, indent=4, ensure_ascii=False)

    with open(os.path.join(log_path, "logs.json"), 'w') as f:
                json.dump(log, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--exp_name", type=str, default="mol-vl_7b")
    parser.add_argument("--tasks", type=str, default="trans_iupac, trans_smiles, general_desp, struct_cap")

    args = parser.parse_args()

    inference(api_url=args.api_url, exp_name=args.exp_name, \
              tasks=args.tasks.replace(" ", "").split(","))