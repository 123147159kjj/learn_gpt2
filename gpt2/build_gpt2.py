import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


# 定义因果自我注意力机制类
class CausalSelfAttention(nn.Module):

    # 初始化函数，接收配置参数config
    def __init__(self, config):
        super().__init__()
        # 确认嵌入维度能被头数整除
        assert config.n_embd % config.n_head == 0
        # 创建线性层用于生成query, key, value向量
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出线性层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 保存配置中的头数和嵌入维度
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 创建一个下三角矩阵作为mask，用于实现因果注意力机制
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    # 前向传播函数
    def forward(self, x):
        B, T, C = x.size()  # 获取输入张量的批大小、序列长度和嵌入维度
        # 计算所有头的query, key, value并调整维度
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 调整维度，使得头成为新的批次维度
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 计算注意力得分，并应用mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # 应用注意力得分到value向量
        y = att @ v
        # 将所有头的输出重新拼接在一起
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出投影层
        y = self.c_proj(y)
        return y


# 定义多层感知机类
class MLP(nn.Module):

    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 输入线性层，输出维度是输入的四倍
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GeLU激活函数
        self.gelu = nn.GELU(approximate='tanh')
        # 输出线性层
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    # 前向传播函数
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


# 定义Transformer的基本块类
class Block(nn.Module):

    # 初始化函数
    def __init__(self, config):
        super().__init__()
        # 层归一化层
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 因果自我注意力机制
        self.attn = CausalSelfAttention(config)
        # 第二个层归一化层
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 多层感知机
        self.mlp = MLP(config)

    # 前向传播函数
    def forward(self, x):
        # 残差连接和子层
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# 使用 @dataclass 装饰器定义一个数据类，用于存储 GPT 模型的配置参数
@dataclass
class GPTConfig:
    # 序列块的最大长度。这是模型可以处理的输入序列的最大长度。
    block_size: int = 1024

    # 词汇表大小。节GPT 使用字对编码 (BPE) 将文本分割成子词。这个值包括了所有的子词合并（50,000 个），
    # 以及额外的字节级标记（256 个），再加上一个特殊标记 ，用于表示文本的结束。
    vocab_size: int = 50257

    # 变量 n_layer 表示模型中 Transformer 块的数量。每个 Transformer 块包含一个多头注意力层和一个前馈神经网络。
    n_layer: int = 12

    # n_head 是多头注意力机制中的头数。多个头允许模型在不同的表示子空间中并行关注信息。
    n_head: int = 12

    # n_embd 是嵌入维度，即单词或序列元素的向量表示的大小。这也是每个 Transformer 层输出向量的维度。
    n_embd: int = 768


# 定义 GPT 模型类，继承自 PyTorch 的 nn.Module
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()  # 调用父类构造函数
        self.config = config  # 设置模型配置

        # 创建一个包含 GPT 模型组件的字典，使用 ModuleDict 来管理
        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入层
            'wpe': nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入层
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer 块列表
            'ln_f': nn.LayerNorm(config.n_embd),  # 最终的层归一化
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # 语言模型的线性头部

    @classmethod
    def from_pretrained(cls, model_type):
        """从预训练的 GPT-2 模型加载权重,加载已经训练好的权重主要是模仿gpt2功能"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"正在加载预训练的 GPT 模型权重：{model_type}")

        # 根据模型类型确定层数、注意力头数和嵌入维度
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }[model_type]
        config_args['vocab_size'] = 50257  # 对于 GPT 模型，词汇表大小始终为 50257
        config_args['block_size'] = 1024  # 对于 GPT 模型，块大小始终为 1024

        # 创建一个从零开始初始化的 minGPT 模型实例
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()  # 获取模型的状态字典 其实就是查看模型结构
        sd_keys = list(sd.keys())  # 获取状态字典的所有键
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # 排除注意力掩码，因为它不是参数

        # 初始化一个来自 Hugging Face 的预训练模型实例
        from transformers import GPT2LMHeadModel
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()  # 获取预训练模型的状态字典

        # 确保所有参数名称和形状匹配
        sd_keys_hf = list(sd_hf.keys())
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # 忽略这些缓冲区
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # 同样，只是掩码（缓冲区）

        # 这些权重在导入时需要转置，因为 OpenAI 的检查点使用的是 "Conv1D" 模块，而我们使用的是标准的线性层
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # 确认两个模型的键数量一致
        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配：{len(sd_keys_hf)} != {len(sd_keys)}"

        # 复制权重
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 对需要转置的 Conv1D 权重进行特殊处理
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():  # 不计算梯度
                    sd[k].copy_(sd_hf[k].t())  # 转置并复制权重
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])  # 复制权重

        return model  # 返回加载了预训练权重的模型实例


# -----------------------------------------------------------------------------
model = GPT.from_pretrained('gpt2')
print("run success!")
