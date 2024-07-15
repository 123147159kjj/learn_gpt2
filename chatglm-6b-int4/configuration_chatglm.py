""" ChatGLM model configuration """

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class ChatGLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~ChatGLMModel`].
    It is used to instantiate an ChatGLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the ChatGLM-6B [THUDM/ChatGLM-6B](https://huggingface.co/THUDM/chatglm-6b) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 150528):
            Vocabulary size of the ChatGLM-6B model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`~ChatGLMModel`] or
            [`~TFChatGLMModel`].
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        inner_hidden_size (`int`, *optional*, defaults to 16384):
            Dimension of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        max_sequence_length (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with.
            Typically set this to something large just in case (e.g., 512 or 1024 or 2048).
        layernorm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon used by the layer normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model should return the last key/values attentions (not used by all models).
        Example:

    ```python
    >>> from configuration_chatglm import ChatGLMConfig
    >>> from modeling_chatglm import ChatGLMModel

    >>> # Initializing a ChatGLM-6B THUDM/ChatGLM-6B style configuration
    >>> configuration = ChatGLMConfig()

    >>> # Initializing a model from the THUDM/ChatGLM-6B style configuration
    >>> model = ChatGLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""
    model_type = "chatglm"

    def __init__(
            self,
            vocab_size=150528,  # 词汇表的大小，即模型能够处理的不同单词或标记的数量。在这个例子中，vocab_size 是 150,528。
            hidden_size=4096,  # 隐藏层的维度，即每个隐藏状态的向量长度。对于 `chatglm`，其值为 4,096。
            num_layers=28,  # 模型中编码器或解码器层的数量。在这个例子中，有 28 层。chatglm采用编解码器结构
            num_attention_heads=32,  # 多头注意力机制中的头数。`chatglm` 使用了 32 个注意力头。
            layernorm_epsilon=1e-5,  # 层归一化（Layer Normalization）操作中的一个非常小的值，用于避免除零错误和数值稳定性问题。
            use_cache=False,  # 如果设置为 `True`，则模型在生成文本时会使用缓存的中间结果，以提高序列生成的效率。
            bos_token_id=150004,  # 这些是特殊令牌的ID，分别代表开始令牌（Beginning Of Sentence）、
            # 结束令牌（End Of Sentence）、掩码令牌、全局掩码令牌以及填充令牌。这些特殊令牌对模型的训练和推理非常重要。
            eos_token_id=150005,
            mask_token_id=150000,
            gmask_token_id=150001,
            pad_token_id=0,
            max_sequence_length=2048,  # 输入序列的最大长度，`chatglm` 可以处理的最长序列长度为 2,048。
            inner_hidden_size=16384,  # 内部隐藏层的尺寸，通常用于前馈网络（Feed Forward Network）内部的线性变换，这里是 16,384。
            position_encoding_2d=True, # 是否使用二维位置编码，这是一种改进的位置编码方法，可以更好地捕捉序列中元素之间的相对位置信息。
            quantization_bit=0,
            quantization_embeddings=False,
            pre_seq_len=None,
            prefix_projection=False,
            **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.layernorm_epsilon = layernorm_epsilon
        self.inner_hidden_size = inner_hidden_size
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.gmask_token_id = gmask_token_id
        self.position_encoding_2d = position_encoding_2d
        self.quantization_bit = quantization_bit
        self.quantization_embeddings = quantization_embeddings
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
