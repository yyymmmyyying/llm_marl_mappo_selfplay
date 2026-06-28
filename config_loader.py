import os
import yaml
from typing import Dict, Any, Optional
import argparse

def load_yaml(file_path: str) -> Dict[str, Any]:
    """加载单个yaml配置文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件不存在: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置，override覆盖base"""
    merged = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_config(merged[k], v)
        else:
            merged[k] = v
    return merged

def load_all_configs(
    train_cfg_path: str,
    infer_cfg_path: str,
    sim_cfg_path: str,
    sp_cfg_path: str,
    reward_cfg_path: str,
    cmd_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """加载全部模块配置并合并"""
    all_cfg = {}
    all_cfg["train"] = load_yaml(train_cfg_path)
    all_cfg["inference"] = load_yaml(infer_cfg_path)
    all_cfg["simulator"] = load_yaml(sim_cfg_path)
    all_cfg["self_play"] = load_yaml(sp_cfg_path)
    all_cfg["reward"] = load_yaml(reward_cfg_path)

    if cmd_override is not None:
        all_cfg = merge_config(all_cfg, cmd_override)
    return all_cfg

def parse_cli_args() -> Dict[str, Any]:
    """解析命令行覆盖参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="覆盖学习率")
    parser.add_argument("--batch_size", type=int, help="覆盖batch size")
    parser.add_argument("--self_play_enable", type=bool, help="开启/关闭自博弈")
    args, _ = parser.parse_known_args()
    override = {}
    if args.lr is not None:
        override["train"] = {"training": {"lr": args.lr}}
    if args.batch_size is not None:
        override["train"] = {"training": {"batch_size": args.batch_size}}
    if args.self_play_enable is not None:
        override["simulator"] = {"self_play": {"enable": args.self_play_enable}}
    return override

if __name__ == "__main__":
    # 示例：加载单卡训练全套配置
    root = os.path.dirname(__file__)
    cfg = load_all_configs(
        train_cfg_path=os.path.join(root, "configs", "train_single.yaml"),
        infer_cfg_path=os.path.join(root, "configs", "inference_vllm.yaml"),
        sim_cfg_path=os.path.join(root, "configs", "simulator.yaml"),
        sp_cfg_path=os.path.join(root, "configs", "self_play.yaml"),
        reward_cfg_path=os.path.join(root, "configs", "reward.yaml"),
        cmd_override=parse_cli_args()
    )
    print("配置加载完成，示例训练学习率：", cfg["train"]["training"]["lr"])