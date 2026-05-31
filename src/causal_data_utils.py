from typing import Dict, List, Tuple, Optional
from src.parameter import DesignPoint
import random
import numpy as np

def parse_key_to_design_point(key: str) -> DesignPoint:
    if key.startswith('lv2.'):
        key = key[4:]
    elif key.startswith('lv1.'):
        key = key[4:]
    point = {}
    if not key:
        return point
    parts = key.split('.')
    for part in parts:
        if '-' in part:
            pragma_id, value = part.rsplit('-', 1)
            try:
                value = int(value)
            except ValueError:
                if value == 'NA':
                    value = ''
                pass
            point[pragma_id] = value
    return point

def enumerate_hamming_one_pair_indices(points: List[Optional[DesignPoint]]) -> List[Tuple[int, int]]:
    n = len(points)
    out: List[Tuple[int, int]] = []
    for i in range(n):
        pi = points[i]
        if pi is None:
            continue
        for j in range(i + 1, n):
            pj = points[j]
            if pj is None:
                continue
            if hamming_distance(pi, pj) == 1:
                out.append((i, j))
    return out

def hamming_distance(point1: DesignPoint, point2: DesignPoint) -> int:
    all_keys = set(point1.keys()) | set(point2.keys())
    distance = 0
    for key in all_keys:
        val1 = point1.get(key, None)
        val2 = point2.get(key, None)
        if val1 != val2:
            distance += 1
    return distance

def create_intervention_pairs(points: List[DesignPoint], max_distance: int=2, num_pairs: Optional[int]=None, seed: int=42) -> List[Tuple[DesignPoint, DesignPoint]]:
    random.seed(seed)
    np.random.seed(seed)
    pairs = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i >= j:
                continue
            dist = hamming_distance(p1, p2)
            if 1 <= dist <= max_distance:
                pairs.append((p1, p2))
    if num_pairs is not None and len(pairs) > num_pairs:
        pairs = random.sample(pairs, num_pairs)
    return pairs

def create_pairs_from_keys(keys: List[str], max_distance: int=2, num_pairs: Optional[int]=None, seed: int=42) -> List[Tuple[str, str, DesignPoint, DesignPoint]]:
    points_dict = {}
    for key in keys:
        point = parse_key_to_design_point(key)
        points_dict[key] = point
    pairs = []
    key_list = list(points_dict.keys())
    for i, key1 in enumerate(key_list):
        for j, key2 in enumerate(key_list):
            if i >= j:
                continue
            point1 = points_dict[key1]
            point2 = points_dict[key2]
            dist = hamming_distance(point1, point2)
            if 1 <= dist <= max_distance:
                pairs.append((key1, key2, point1, point2))
    if num_pairs is not None and len(pairs) > num_pairs:
        random.seed(seed)
        pairs = random.sample(pairs, num_pairs)
    return pairs