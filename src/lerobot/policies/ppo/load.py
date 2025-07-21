import torch.nn as nn
import torch

# 加载pt文件
pt_path = "/home/yk/lerobot/src/lerobot/policies/ppo/2025-07-16_11-55-39_ppo_torch/checkpoints/agent_72000.pt"
modules = torch.load(pt_path, map_location="cpu")  # 这里可以根据需要修改map_location

# 构建policy和value网络结构
policy = nn.Sequential(
    nn.Linear(2018, 512, bias=True),
    nn.ELU(alpha=1.0),
    nn.Linear(512, 256, bias=True),
    nn.ELU(alpha=1.0),
    nn.Linear(256, 128, bias=True),
    nn.ELU(alpha=1.0),
    nn.Linear(128, 64, bias=True),
    nn.ELU(alpha=1.0)
)
policy_layer = nn.Linear(64, 6, bias=True)

value = nn.Sequential(
    nn.Linear(2018, 512, bias=True),
    nn.ELU(alpha=1.0),
    nn.Linear(512, 256, bias=True),
    nn.ELU(alpha=1.0),
    nn.Linear(256, 128, bias=True),
    nn.ELU(alpha=1.0),
    nn.Linear(128, 64, bias=True),
    nn.ELU(alpha=1.0)
)
value_layer = nn.Linear(64, 1, bias=True)

model = {
    "net_container": policy,
    "policy_layer": policy_layer,
    "value_container": value,
    "value_layer": value_layer
}

# 正确加载pt文件中的参数到model
if isinstance(modules, dict):
    # 兼容skrl保存的agent字典结构
    # 1. policy网络参数
    if "policy" in modules and hasattr(modules["policy"], "state_dict"):
        # skrl保存的policy是模型对象
        policy_state = modules["policy"].state_dict()
        # 只加载前面部分
        policy.load_state_dict({k.replace("net_container.", ""): v for k, v in policy_state.items() if "net_container" in k})
        policy_layer.load_state_dict({k.replace("policy_layer.", ""): v for k, v in policy_state.items() if "policy_layer" in k})
    elif "policy" in modules and isinstance(modules["policy"], dict):
        # 直接是state_dict
        policy.load_state_dict({k.replace("net_container.", ""): v for k, v in modules["policy"].items() if "net_container" in k})
        policy_layer.load_state_dict({k.replace("policy_layer.", ""): v for k, v in modules["policy"].items() if "policy_layer" in k})

    # 2. value网络参数
    if "value" in modules and hasattr(modules["value"], "state_dict"):
        value_state = modules["value"].state_dict()
        value.load_state_dict({k.replace("net_container.", ""): v for k, v in value_state.items() if "net_container" in k})
        value_layer.load_state_dict({k.replace("value_layer.", ""): v for k, v in value_state.items() if "value_layer" in k})
    elif "value" in modules and isinstance(modules["value"], dict):
        value.load_state_dict({k.replace("net_container.", ""): v for k, v in modules["value"].items() if "net_container" in k})
        value_layer.load_state_dict({k.replace("value_layer.", ""): v for k, v in modules["value"].items() if "value_layer" in k})
