
from typing import Dict, List, Tuple

import torch.nn as nn

class StageModule(nn.Module):
    @property
    def keys_from_source(self) -> List[str]:
        raise NotImplementedError

    @property
    def sizes_from_prev_stage(self) -> Dict[str, Tuple]:
        raise NotImplementedError

    @property
    def sizes_for_next_stage(self) -> Dict[str, Tuple]:
        raise NotImplementedError
    
    @property
    def layers(self) -> List[nn.Module]:
        raise NotImplementedError
