import torch
import torch.nn as nn

class SequentialModel(nn.Module):
    """
    基于给定架构的Sequential模型:
    Sequential(
        (0): Linear(in_features=2018, out_features=512, bias=True)
        (1): ELU(alpha=1.0)
        (2): Linear(in_features=512, out_features=256, bias=True)
        (3): ELU(alpha=1.0)
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ELU(alpha=1.0)
        (6): Linear(in_features=128, out_features=64, bias=True)
        (7): ELU(alpha=1.0)
    )
    """
    
    def __init__(self, input_features=2018, output_features=64):
        super(SequentialModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_features, 512, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(512, 256, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(256, 128, bias=True),
            nn.ELU(alpha=1.0),
            nn.Linear(128, output_features, bias=True),
            nn.ELU(alpha=1.0)
        )
    
    def forward(self, x):
        return self.network(x)

# 创建模型实例
def create_sequential_model(input_features=2018, output_features=64, device="cuda"):
    """
    创建并返回Sequential模型实例
    
    Args:
        input_features (int): 输入特征维度，默认2018
        output_features (int): 输出特征维度，默认64
        device (str): 设备类型，默认"cuda"
    
    Returns:
        SequentialModel: 配置好的模型实例
    """
    model = SequentialModel(input_features, output_features)
    model = model.to(device)
    return model

if __name__ == "__main__":
    # 测试模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_sequential_model(device=device)
    
    # 打印模型架构
    print("模型架构:")
    print(model)
    
    # 测试前向传播
    test_input = torch.randn(1, 2018).to(device)
    output = model(test_input)
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}") 