
from transformers import LlamaConfig

from ..configuration_live import LiveConfigMixin

class LiveLlamaConfig(LlamaConfig, LiveConfigMixin):
    mask_threshold: float = 0.5
    loss_threshold: float = 0.6



class LiveLlamaConfigMamba(LiveLlamaConfig):
    vison_loss: bool = True
    
class LiveLlamaConfigMambaFt(LiveLlamaConfig):
    vison_loss: bool = False