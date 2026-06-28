import requests
import json
from typing import Optional
from config_loader import load_all_configs, parse_cli_args
import os

root = os.path.dirname(os.path.dirname(__file__))
cfg = load_all_configs(
    train_cfg_path=os.path.join(root, "configs", "train_single.yaml"),
    infer_cfg_path=os.path.join(root, "configs", "inference_vllm.yaml"),
    sim_cfg_path=os.path.join(root, "configs", "simulator.yaml"),
    sp_cfg_path=os.path.join(root, "configs", "self_play.yaml"),
    reward_cfg_path=os.path.join(root, "configs", "reward.yaml"),
    cmd_override=parse_cli_args()
)
INFER_CFG = cfg["inference"]
SERVER_URL = f"http://{INFER_CFG['server']['host']}:{INFER_CFG['server']['port']}/generate"

def generate(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "temperature": INFER_CFG["generation"]["temperature"],
        "top_p": INFER_CFG["generation"]["top_p"],
        "max_new_tokens": INFER_CFG["generation"]["max_new_tokens"]
    }
    resp = requests.post(SERVER_URL, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["output"]

def planner_agent(query: str) -> str:
    template = INFER_CFG["prompt"]["planner_template"]
    prompt = template.format(query=query)
    return generate(prompt)

def code_agent(step_desc: str) -> str:
    template = INFER_CFG["prompt"]["code_template"]
    prompt = template.format(step_desc=step_desc)
    return generate(prompt)

def judge_agent(origin_task: str, code_output: str) -> str:
    template = INFER_CFG["prompt"]["judge_template"]
    prompt = template.format(origin_task=origin_task, code_output=code_output)
    return generate(prompt)

if __name__ == "__main__":
    task = "读取csv文件计算平均值"
    plan = planner_agent(task)
    print("规划智能体输出：\n", plan)