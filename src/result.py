from src.parameter import DesignPoint
from enum import Enum
from typing import Dict, Optional

class Result(object):

    class RetCode(Enum):
        PASS = 0
        UNAVAILABLE = -1
        EARLY_REJECT = -2
        DUPLICATED = -3

    def __init__(self, ret_code_str: str='PASS'):
        self.point: Optional[DesignPoint] = None
        self.ret_code: Result.RetCode = self.RetCode[ret_code_str]
        self.valid: bool = True
        self.quality: float = -float('inf')
        self.perf: float = 0.0
        self.actual_perf: float = 0.0
        self.area: float = -float('inf')
        self.res_util: Dict[str, float] = {'util-BRAM': 0, 'util-DSP': 0, 'util-LUT': 0, 'util-FF': 0, 'total-BRAM': 0, 'total-DSP': 0, 'total-LUT': 0, 'total-FF': 0}
        self.eval_time: float = 0.0