import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import tiktoken


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
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        # idx 是输入的索引张量，形状为 (B, T)，其中 B 表示批量大小，T 表示序列长度
        B, T = idx.size()

        # 断言检查，确保序列长度不超过模型配置的块大小（block_size）
        assert T <= self.config.block_size, f"无法对长度为{T}的序列进行前向传播，因为块大小仅为{self.config.block_size}"

        # 创建位置编码张量，范围从0到T-1
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # 形状为 (T)

        # 通过Transformer的位置嵌入层获取位置编码，形状为 (T, n_embd)
        pos_emb = self.transformer.wpe(pos)

        # 通过Transformer的词嵌入层获取词嵌入，形状为 (B, T, n_embd)
        tok_emb = self.transformer.wte(idx)

        # 将词嵌入与位置编码相加
        x = tok_emb + pos_emb

        # 依次通过Transformer中的每一个Block进行前向传播
        for block in self.transformer.h:
            x = block(x)

        # 通过Transformer的最终层归一化层
        x = self.transformer.ln_f(x)

        # 最后通过语言模型的分类器（即全连接层），输出形状为 (B, T, vocab_size)，即每个位置上的词汇预测概率
        logits = self.lm_head(x)

        # 预测的logits（模型的原始输出，通常是未经过softmax函数的值）和真实的目标值
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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
class DataLoaderLite:
    def __init__(self, B, T):
        # 构造函数初始化类的属性
        self.B = B  # 批量大小
        self.T = T  # 序列长度

        # 在初始化时，从磁盘加载tokens到内存中
        with open('input.txt', 'r') as f:
            text = f.read()  # 读取整个文本文件的内容
        enc = tiktoken.get_encoding('gpt2')  # 创建GPT-2的编码器实例
        tokens = enc.encode(text)  # 将文本转换为tokens
        self.tokens = torch.tensor(tokens)  # 将tokens列表转换为PyTorch的张量
        print(f"已加载 {len(self.tokens)} 个tokens")  # 打印加载的tokens总数
        print(f"1轮 = {len(self.tokens) // (B * T)} 个批次")  # 计算并打印一轮包含的批次数量

        # 初始化状态
        self.current_position = 0  # 当前位置指针，用于跟踪数据流

    def next_batch(self):
        B, T = self.B, self.T  # 提取批量大小和序列长度

        # 从当前位置开始，截取一个批次的数据
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]

        # 分割数据为输入和目标
        x = (buf[:-1]).view(B, T)  # 输入数据，形状为 (B, T)
        y = (buf[1:]).view(B, T)  # 目标数据，形状为 (B, T)

        # 更新当前位置指针
        self.current_position += B * T

        # 如果加载下一个批次会超出tokens的边界，则重置位置指针
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        # 返回一个批次的输入和目标数据
        return x, y

# --------------------------------测试--------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

train_loader = DataLoaderLite(B=4, T=32)

# get logits
model = GPT(GPTConfig())
model.to(device)
# optimize!
# 创建AdamW优化器实例，传入模型的所有可训练参数，设置学习率为3e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 开始训练循环，迭代50次
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    # 清零梯度，防止梯度累积
    optimizer.zero_grad()

    # 前向传播：将输入数据x送入模型，得到模型输出logits和损失loss
    logits, loss = model(x, y)

    # 反向传播：计算损失相对于模型参数的梯度
    loss.backward()

    # 更新参数：根据计算出的梯度和优化算法更新模型参数
    optimizer.step()

    # 打印当前迭代次数和损失值
    print(f"步骤 {i}，损失: {loss.item()}")

sys.exit(0)
# prefix tokens
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
x = tokens.to(device)

# 开始生成！此时x的形状是(B, T)，其中B=5（batch size），T=8（初始token序列长度）
# 为实验设置随机种子，确保每次运行的结果一致
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 循环直到生成的序列长度达到设定的最大值max_length
while x.size(1) < max_length:
    # 前向传播模型获取logits（未归一化的概率）
    with torch.no_grad():  # 不需要计算梯度，节省内存
        logits = model(x)  # 输出形状为 (B, T, vocab_size)

    # 取出logits的最后一位置，即最后一个token的logits
    logits = logits[:, -1, :]  # 形状变为 (B, vocab_size)

    # 转换logits为概率分布
    probs = F.softmax(logits, dim=-1)

    # 进行top-k采样，选择前50个最高概率的token（这是Hugging Face管道的默认设置）
    # topk_probs形状变为 (B, 50)，topk_indices也是 (B, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

    # 从top-k的概率中选择一个token
    # 注意：multinomial函数不需要输入概率总和为1
    ix = torch.multinomial(topk_probs, 1)  # 形状为 (B, 1)

    # 根据选择的索引，获取对应的token
    xcol = torch.gather(topk_indices, -1, ix)  # 形状为 (B, 1)

    # 将新选出的token附加到序列末尾
    x = torch.cat((x, xcol), dim=1)

# 打印生成的文本
for i in range(num_return_sequences):
    # 获取第i个序列的token，并转换为list
    tokens = x[i, :max_length].tolist()
    # 解码token列表为文本字符串
    decoded = enc.decode(tokens)
    # 打印生成的文本
    print(">", decoded)
# ------生面演示过程:自定义模型-->加载开源模型参数-->输入(编码)-->模型处理-->输出(解码)----
