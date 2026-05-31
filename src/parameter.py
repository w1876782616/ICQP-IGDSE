import ast
from typing import Dict, List, Optional, Tuple, Type, Union, Set

class DesignParameter(object):

    def __init__(self, name: str=''):
        self.name: str = name
        self.default: Union[str, int] = 1
        self.option_expr: str = ''
        self.scope: List[str] = []
        self.order: Dict[str, str] = {}
        self.deps: List[str] = []
        self.child: List[str] = []
        self.value: Union[str, int] = 1
DesignSpace = Dict[str, DesignParameter]
DesignPoint = Dict[str, Union[int, str]]

def gen_key_from_design_point(point: DesignPoint) -> str:

    def _fmt_val(v) -> str:
        if v is None:
            return 'NA'
        return str(v)
    return '.'.join([f'{pid}-{_fmt_val(point.get(pid, None))}' for pid in sorted(point.keys())])

def check_option_syntax(option_expr: str, log) -> Tuple[bool, List[str]]:
    try:
        stree = ast.parse(option_expr)
    except SyntaxError:
        log.error(f'"options" error: Illegal option list {option_expr}')
        return (False, [])
    names = set()
    iter_val = None
    for node in ast.walk(stree):
        if isinstance(node, ast.ListComp):
            funcs = [n.func.id for n in ast.walk(node.elt) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            elt_vals = [n.id for n in ast.walk(node.elt) if isinstance(n, ast.Name) and n.id not in funcs and (n.id != '_')]
            assert len(elt_vals) <= 1, 'Found more than one iterators in {0}'.format(option_expr)
            if len(elt_vals) == 1:
                iter_val = elt_vals[0]
        elif isinstance(node, ast.Name):
            names.add(node.id)
    if iter_val:
        names.remove(iter_val)
    for ptype in ['int', 'str', 'float']:
        if ptype in names:
            names.remove(ptype)
    return (True, list(names))

def check_order_syntax(order_expr: str, log) -> Tuple[bool, str]:
    try:
        stree = ast.parse(order_expr)
    except SyntaxError:
        log.error(f'"order" error: Illegal order expression {order_expr}')
        return (False, '')
    names = set()
    for node in ast.walk(stree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    if len(names) != 1:
        log.error(f'"order" should have one and only one variable in {order_expr} but found {len(names)}')
        return (False, '')
    return (True, names.pop())

def create_design_parameter(param_id: str, ds_config: Dict[str, Union[str, int]], param_cls: Type[DesignParameter], log) -> Optional[DesignParameter]:
    if param_cls == DesignParameter:
        param = DesignParameter(param_id)
        if 'ds_type' not in ds_config:
            log.warning(f'Missing attribute "ds_type" in {param_id}. Some optimization may not be triggered')
        else:
            param.ds_type = str(ds_config['ds_type']).upper()
    else:
        log.error('Unrecognized parameter type')
        return None
    if 'options' not in ds_config:
        log.error('Missing attribute "options" in %s', param_id)
        return None
    if 'TIL-' in param.ds_type:
        param.option_expr = str([1])
    else:
        param.option_expr = str(ds_config['options'])
    check, param.deps = check_option_syntax(param.option_expr, log)
    if not check:
        return None
    if 'order' in ds_config:
        check, var = check_order_syntax(str(ds_config['order']), log)
        if not check:
            log.warning(f'Failed to parse "order" of {param_id}, ignore.')
        else:
            param.order = {'expr': str(ds_config['order']), 'var': var}
    if 'default' not in ds_config:
        log.error(f'Missing attribute "default" in {param_id}')
        return None
    param.default = ds_config['default']
    return param

def get_default_point(ds: DesignSpace) -> DesignPoint:
    point: DesignPoint = {}
    for pid, param in ds.items():
        point[pid] = param.default
    return point

def check_design_space(params: DesignSpace, log) -> int:
    error = 0
    for pid, param in params.items():
        has_error = False
        for dep in param.deps:
            if dep == pid:
                log.error(f'Parameter {pid} cannot depend on itself')
                error += 1
                has_error = True
            if dep not in params.keys():
                log.error(f'Parameter {pid} depends on {dep} which is undefined or not allowed')
                error += 1
                has_error = True
        if has_error:
            continue
        local = {}
        for dep in param.deps:
            local[dep] = params[dep].default
        options: Optional[List[Union[int, str]]] = None
        try:
            options = eval(param.option_expr, local)
        except (NameError, ValueError, TypeError, ZeroDivisionError) as err:
            log.error('Failed to get the options of parameter %s: %s', pid, str(err))
            error += 1
        if options is not None and param.order and isinstance(param, DesignParameter):
            for option in options:
                if eval(param.order['expr'], {param.order['var']: option}) is None:
                    log.error('Failed to evaluate the order of option %s in parameter %s', option, pid)
                    error += 1
    return error

def analyze_child_in_design_space(params: DesignSpace) -> None:
    for pid, param in params.items():
        for dep in param.deps:
            params[dep].child.append(pid)
    for param in params.values():
        param.child = list(dict.fromkeys(param.child))

def topo_sort_param_ids(space: DesignSpace) -> List[str]:

    def helper(curr_id: str, visited: Set[str], stack: List[str]) -> None:
        visited.add(curr_id)
        for dep in space[curr_id].deps:
            if dep not in visited:
                helper(dep, visited, stack)
        stack.append(curr_id)
    visited: Set[str] = set()
    stack: List[str] = []
    for pid in space.keys():
        if pid not in visited:
            helper(pid, visited, stack)
    return stack

def count_design_points(ds: DesignSpace, log) -> int:

    def helper(ds: DesignSpace, sorted_ids: List[str], idx: int, point: DesignPoint) -> int:
        if idx == len(sorted_ids):
            return 1
        pid = sorted_ids[idx]
        param = ds[pid]
        options = eval(param.option_expr, point)
        counter = 0
        if param.child:
            for option in options:
                point[pid] = option
                counter += helper(ds, sorted_ids, idx + 1, point)
        else:
            counter = len(options) * helper(ds, sorted_ids, idx + 1, point)
        log.debug(f'Node {pid}: {counter}')
        return counter
    point = get_default_point(ds)
    sorted_ids = topo_sort_param_ids(ds)
    return helper(ds, sorted_ids, 0, point)

def compile_design_space(user_ds_config: Dict[str, Dict[str, Union[str, int]]], scope_map: Optional[Dict[str, List[str]]], log) -> Optional[DesignSpace]:
    params: Dict[str, DesignParameter] = {}
    for param_id, param_config in user_ds_config.items():
        param = create_design_parameter(param_id, param_config, DesignParameter, log)
        if param:
            if param.ds_type not in ['PARALLEL', 'PIPELINE', 'TILING', 'TILE']:
                param.scope.append('GLOBAL')
            elif scope_map and param_id in scope_map:
                param.scope = scope_map[param_id]
            params[param_id] = param
    error = check_design_space(params, log)
    if error > 0:
        log.error(f'Design space has {error} errors')
        return None
    analyze_child_in_design_space(params)
    num_ds = count_design_points(params, log)
    log.info(f'Design space contains {num_ds} valid design points')
    log.info('Finished design space compilation')
    return (params, num_ds)