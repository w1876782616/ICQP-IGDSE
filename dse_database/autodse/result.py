from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Union
DesignPoint = Dict[str, Union[int, str]]

class Job(object):

    class Status(Enum):
        INIT = 0
        APPLIED = 1

    def __init__(self, path: str):
        self.path: str = path
        self.key: str = 'NotAPPLIED'
        self.point: Optional[DesignPoint] = None
        self.status: Job.Status = Job.Status.INIT

class Result(object):

    class RetCode(Enum):
        PASS = 0
        UNAVAILABLE = -1
        ANALYZE_ERROR = -2
        EARLY_REJECT = -3
        TIMEOUT = -4
        DUPLICATED = -5

    def __init__(self, ret_code_str: str='PASS'):
        self.point: Optional[DesignPoint] = None
        self.ret_code: Result.RetCode = self.RetCode[ret_code_str]
        self.valid: bool = False
        self.path: Optional[str] = None
        self.quality: float = -float('inf')
        self.perf: float = 0.0
        self.res_util: Dict[str, float] = {'util-BRAM': 0, 'util-DSP': 0, 'util-LUT': 0, 'util-FF': 0, 'total-BRAM': 0, 'total-DSP': 0, 'total-LUT': 0, 'total-FF': 0}
        self.eval_time: float = 0.0

class MerlinResult(Result):

    def __init__(self, ret_code_str: str='PASS'):
        super(MerlinResult, self).__init__(ret_code_str)
        self.criticals: List[str] = []
        self.code_hash: Optional[str] = None

class HierPathNode(NamedTuple):
    nid: str
    latency: float
    is_compute_bound: bool

class HLSResult(Result):

    def __init__(self, ret_code_str: str='PASS'):
        super(HLSResult, self).__init__(ret_code_str)
        self.ordered_paths: Optional[List[List[HierPathNode]]] = None

class BitgenResult(Result):

    def __init__(self, ret_code_str: str='PASS'):
        super(BitgenResult, self).__init__(ret_code_str)
        self.freq: float = 0.0