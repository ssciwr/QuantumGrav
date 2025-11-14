import yaml
import copy
import importlib
from typing import Any, Tuple, Dict
import numpy as np

from . import utils


def sweep_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Dict[str, Any]:
    """Constructor for the !sweep yaml tag, which in a yaml file can look like this:
    tagname: !sweep
        values: [a,b,c...]
    Args:
        loader (yaml.SafeLoader): loader that loads the yaml file
        node (yaml.nodes.MappingNode): the current !sweep node to process

    Returns:
        Dict[str, Any]: dictionary of the form {type:sweep, values: [v1,v2,v3,...]}
    """
    values = loader.construct_sequence(node.value[0][1])
    return {"type": "sweep", "values": values}


def coupled_sweep_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Dict[str, Any]:
    """Constructor for the !coupled-sweep yaml tag which is designed to tie a sequence of values
    to a !sweep tag such that they proceed in lockstep like a 'zip' operation. Example:
    tagname: !sweep
        values: [a,b,c...]
    coupled: !coupled-sweep
        target: path.in.config.to.tagname
        values: [1,2,3,...]
    Args:
        loader (yaml.SafeLoader): loader that loads the yaml file
        node (yaml.nodes.MappingNode): the current !coupled-sweep node to process

    Returns:
        Dict[str, Any]: dictionary of the form {keys:'type', type: 'coupled-sweep', values: [v1,v2,v3,...], target: [path, to, target]}
    """
    tokens = node.value[0][1].value.split(".")
    sweep_target = []
    for token in tokens:
        split_token = token.split("[")
        if len(split_token) > 1:
            index = int(split_token[1][:-1])
            sweep_target.append(split_token[0])
            sweep_target.append(index)
        else:
            sweep_target.append(token)

    values = loader.construct_sequence(node.value[1][1])
    return {"target": sweep_target, "values": values, "type": "coupled-sweep"}


def arange_inclusive(
    start: int | float, stop: int | float, step: int | float
) -> np.ndarray:
    """Return a numpy array from start to stop (inclusive),
    with the given step, working for both int and float.

    Args:
        start (int | float): Start value
        stop (int | float): Stop value (inclusive)
        step (int | float): Step size

    Returns:
        np.ndarray: Numpy array of values from start to stop (inclusive)
            with the given step.
    """
    if step == 0:
        raise ValueError("step must not be zero")

    # Compute number of steps (round to avoid float precision errors)
    # +1 to include stop
    num = int(round((stop - start) / step)) + 1

    # Generate linearly spaced values (always includes stop)
    values = np.linspace(start, start + (num - 1) * step, num)

    # Adjust last value to exactly match stop when it's very close
    if abs(values[-1] - stop) < abs(step) * 1e-9:
        values[-1] = stop

    # If all inputs were ints, cast result back to int
    if all(isinstance(x, int) for x in (start, stop, step)):
        values = values.astype(int)

    return values


def range_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Dict[str, Any]:
    """Constructor for the !range tag:
    tagname: !range
      start: 0
      stop: 10
      step: 2

      or

    tagname: !range
      start: 1.0e-05
      stop: 0.1
      log: true
      size: 5

    In the latter case, sample value in the log domain.
    The 'step' or 'log' field is optional.
    If there is no `size` specified for `log`, use 5 as default value.

    The end value is inclusive when step is given to make it consistent with Optuna sampling.

    Args:
        loader (yaml.SafeLoader): loader that loads the yaml file
        node (yaml.nodes.MappingNode): the current !range node to process

    Returns:
        Dict[str, Any]: dict of the form: {"type": "range",
                                            "values": values as np.array,
                                            "tune_values": Tuple[start, end, step or log]}
    """
    # using PyYAML's mapping constructor to ensure value types are correct
    mapping = loader.construct_mapping(node, deep=True)
    start = mapping.get(
        node.value[0][0].value
    )  # use indices to access keys in case key names change
    end = mapping.get(node.value[1][0].value)
    step_or_log = mapping.get(node.value[2][0].value) if len(node.value) > 2 else None

    # prepare values
    if isinstance(step_or_log, bool):
        size = mapping.get(node.value[3][0].value) if len(node.value) > 3 else 5
        # in case start or end are given as 1e-5 etc., convert to float
        start = float(start)
        end = float(end)
        if step_or_log:  # log = true
            values = np.exp(np.random.uniform(np.log(start), np.log(end), size=size))
        else:  # log = false
            values = np.random.uniform(start, end, size=size)

    elif isinstance(step_or_log, (int, float)):
        values = arange_inclusive(start, end, step_or_log)

    else:  # no step or log specified
        default_step = 1 if isinstance(start, int) and isinstance(end, int) else 0.1
        values = arange_inclusive(start, end, default_step)
        step_or_log = default_step

    return {"type": "range", "values": values, "tune_values": (start, end, step_or_log)}


def reference_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> Dict[str, Any]:
    """Constructor for the !reference tag:

    referred_tag:...
    tagname: !reference
      target: path.to.referred_tag

    This tag is used where value of the referred tag is not fixed at config load time,
    but might be changed later on. This always points to the current value of the referred tag.

    Args:
        loader (yaml.SafeLoader): loader that loads the yaml file
        node (yaml.nodes.MappingNode): the current !reference node to process

    Returns:
        Dict[str, Any]: dict of the form: {"type": "reference", "target": [path, to, referred_tag]}
    """
    tokens = node.value[0][1].value.split(".")
    reference_target = []
    for token in tokens:
        split_token = token.split("[")
        if len(split_token) > 1:
            index = int(split_token[1][:-1])
            reference_target.append(split_token[0])
            reference_target.append(index)
        else:
            reference_target.append(token)

    return {"type": "reference", "target": reference_target}


def object_constructor(loader: yaml.SafeLoader, node: yaml.nodes.ScalarNode) -> Any:
    """Read a type name from a YAML file and return it such that and instance can be build.
    This must give the full name/path of the type, e.g.
    if you do something like:
    ```python
       import torch_geometric.nn as tgnn
    ```
    and you want to get the `SAGEConv` type from torch_geometric, you cannot write

    nodename: !pyobject tgnn.SAGEConv

    but have to write:

    nodename: !pyobject torch_geometric.nn.SAGEConv

    for example. The modules will be reimported.

    Args:
        loader (yaml.SafeLoader): loader that loads the yaml file
        node (yaml.nodes.ScalarNode): The current !pyobject type to be read

    Returns:
        Any: A type or class, e.g., torch_geometric.nn.SAGEConv
    """
    value = node.value
    split_value = value.split(".")
    objectname = split_value[-1]
    modulename = ".".join(split_value[0 : len(split_value) - 1])
    tpe = None
    try:
        module = importlib.import_module(modulename)
    except Exception as e:
        raise ValueError(f"Importing module {modulename} unsuccessful") from e

    try:
        tpe = getattr(module, objectname)
    except Exception as e:
        raise ValueError(f"Could not load name {objectname} from {modulename}") from e

    return tpe


def get_loader():
    """Integrate custom tags into the loader system of the PyYAML library

    Returns:
        loader: yaml.SafeLoader instance
    """
    loader = yaml.SafeLoader
    loader.add_constructor("!sweep", sweep_constructor)
    loader.add_constructor("!coupled-sweep", coupled_sweep_constructor)
    loader.add_constructor("!range", range_constructor)
    loader.add_constructor("!pyobject", object_constructor)
    loader.add_constructor("!reference", reference_constructor)
    return loader


class ConfigHandler:
    """A class for handling the splitting of a configuration file with !sweep and !coupled-sweep
    nodes into a set of different config files according to the cartesian product of
    the sweep dimension and a `zip` operation of the coupled-sweeps with their target nodes.
    The targets have to be given as full paths from the top level of the config
    """

    def __init__(self, config: dict, name_addition="run"):
        """Initialize a new `ConfigHandler` class given a config.

        Args:
            config (dict): Dictionary containing the config data to process, possibly with
            nodes derived from !sweep and !coupled-sweep nodes.
        """
        self.config = config
        self.run_configs = []
        sweep_targets = {}
        coupled_targets = {}

        # make dictionaries that record sweep dims and their coupled partners
        # then assign the sweep partners to their sweep dimensions
        self._extract_sweep_dims([], self.config, sweep_targets, coupled_targets)

        if len(sweep_targets) == 0 and len(coupled_targets) == 0:
            self.run_configs = [
                config,
            ]
        else:
            # postprocess the coupled targets to augment the sweep targets
            for k, v in coupled_targets.items():
                if len(v["values"]) != len(sweep_targets[tuple(v["target"])]["values"]):
                    raise ValueError(
                        f"Incompatible lengths for coupled-sweep {v['target']}"
                    )

                if (
                    "partner" in sweep_targets[tuple(v["target"])]
                    and sweep_targets[tuple(v["target"])]["partner"] is not None
                ):
                    sweep_targets[tuple(v["target"])]["partner"].append(v["values"])
                else:
                    sweep_targets[tuple(v["target"])]["partner"] = [v["values"]]

                if (
                    "partner_path" in sweep_targets[tuple(v["target"])]
                    and sweep_targets[tuple(v["target"])]["partner_path"] is not None
                ):
                    sweep_targets[tuple(v["target"])]["partner_path"].append(v["path"])
                else:
                    sweep_targets[tuple(v["target"])]["partner_path"] = [v["path"]]

            # construct the configs
            self.run_configs = self._construct_run_configs(sweep_targets)

        for i, cfg in enumerate(self.run_configs):
            cfg["model"]["name"] = f"{cfg['model']['name']}_{name_addition}_{i}"

    def _extract_sweep_dims(
        self,
        k: list[str],
        cfg_node: dict[str, Any],
        sweep_targets: dict[Tuple[str, ...], Any],
        coupled_targets: dict[Tuple[str, ...], Any],
    ) -> None:
        """Recursively extract the sweep and coupled sweep dimensions from the supplied config
        and augment the sweep dimension with their coupled partner dimensions and
        the path and possibly path to their partner in the config.

        Args:
            k (list[str]): path to the sweep/coupled-sweep nodes to put together.
            cfg_node (dict[str, Any]): current config node
            sweep_targets (dict[str, Any]): dict to store sweep dimensions in
            coupled_targets (dict[str, Any]): dict to store coupled-sweep dimension in
        """
        # TODO: do we need list support on the outermost level?
        if not isinstance(
            cfg_node, dict
        ):  # only dictionary nodes are interesting by design, others are not capable of storing the data structures needed
            return

        for key, node in cfg_node.items():
            if isinstance(node, dict) and "type" in node:
                if node["type"] == "sweep":
                    # when a sweep dimension is found, we record its path and values
                    # and use the path as a key in the dict. Partner is none for now
                    k.append(key)
                    sweep_targets[tuple(k)] = {
                        "path": copy.deepcopy(k),
                        "values": node["values"],
                        "partner": None,
                    }
                    if len(k) > 0:
                        k.pop()
                elif node["type"] == "coupled-sweep":
                    # when a coupled-sweep dimension is found, we record its path and values
                    # and use the path as a key in the dict. this will later be used to
                    # assign the partners to the sweep dimensions
                    k.append(key)
                    coupled_targets[tuple(k)] = {
                        "path": copy.deepcopy(k),
                        "target": node["target"],
                        "values": node["values"],
                    }
                    if len(k) > 0:
                        k.pop()
                else:
                    # if it's a regular dict node with 'type'is found, we record it's path and go one level down
                    k.append(key)
                    self._extract_sweep_dims(k, node, sweep_targets, coupled_targets)
                    if len(k) > 0:
                        k.pop()
            elif isinstance(node, dict):
                # no type dict node is found we go down one level as well
                k.append(key)
                self._extract_sweep_dims(k, node, sweep_targets, coupled_targets)
                if len(k) > 0:
                    k.pop()
            elif isinstance(node, list):
                # for list nodes, we iterate over them and treat the indices as path elements
                k.append(key)
                for i in range(len(node)):
                    k.append(i)
                    self._extract_sweep_dims(k, node[i], sweep_targets, coupled_targets)
                    if len(k) > 0:
                        k.pop()
                if len(k) > 0:
                    k.pop()
            else:
                # scalar nodes don't count because they are leafs
                pass

    def _construct_cartesian_product(
        self,
        elements: list[Any],
        current_list: list[Any],
        possible_partner: list[Any] | None,
        all_lists: list[Tuple[list[Any], list[Any] | None]],
        *args,
        i=0,
    ) -> None:
        """Recursively construct a list of tuples that represent the cartesian product of the data in a set of lists

        Args:
            elements (list[Any]): list of tuples representing the cartesian product of the data
            current_list (list[Any]): current list of values under consideration
            possible_partner (list[Any] | None): possible coupled-sweep partner list
            all_lists (list[tuple[list[Any], list[Any] | None]]): lists to construct the cartesian product of
            i (int, optional): Index into the `all_lists` argument. Defaults to 0.
        """
        # this function essentially is cpp metaprogramming with python: collect the an element of the cartesian product
        # of 'all_lists' in the arguments, taking care of the alignment of sweep dims and their partners

        # Explanation:
        # i is the index that goes over the set of lists to construct the cartesian product of,
        # e.g. [([1,2,3], [[a,b,c],]), ([3.14, 6.28], None)]. first element in each tuple is the sweep-dim,
        # the second is a list of coupled dims or None if no coupled dims exist. i is the index of the
        # current list under scrutiny, e.g. [12,3].  to build the cartesian product. start with i = 0,
        # current_list = [1,2,3]. it goes through the current list, and for each element accessed by index k a
        # dds the current element to the variadic args and  recursively calls itself with the next list,
        # e.g ([3.14,...], ). If there are couplings, it adds those to the variadic args too. the
        # functoin stops when the index of dims i is exhausted, then it appends the it adds the final element
        # to the var-args and appends the whole var-arg collection it got to the target list 'elements'.
        #  In essence this constructs a series of nested loops:
        # for v1 in list1:
        #     for v2 in list2:
        #         for v3 in list3:
        #             ...
        #             elements.append((v1, v2, v3,...)
        i += 1
        for k in range(len(current_list)):
            v = current_list[k]
            if i < len(all_lists):
                if possible_partner is not None:
                    # possible partner is a coupled sweep dimension
                    # there can be multiple, hence collect values first,
                    # then spread them out
                    w = []
                    for j in range(len(possible_partner)):
                        w.append(possible_partner[j][k])
                    self._construct_cartesian_product(
                        elements,
                        all_lists[i][0],
                        all_lists[i][1],
                        all_lists,
                        *args,
                        v,
                        *w,
                        i=i,
                    )
                else:
                    # without partner -> only add v to args
                    self._construct_cartesian_product(
                        elements,
                        all_lists[i][0],
                        all_lists[i][1],
                        all_lists,
                        *args,
                        v,
                        i=i,
                    )
            else:
                # end of the lists -> final elements. no recursion
                if possible_partner is not None:
                    w = []
                    for j in range(len(possible_partner)):
                        w.append(possible_partner[j][k])
                    elements.append([*args, v, *w])
                else:
                    elements.append([*args, v])

    def _construct_run_configs(
        self, sweep_targets: dict[Tuple[str, ...], Any]
    ) -> list[dict[str, Any]]:
        """Construct the set of run configuration dictionaries from the sweep target dictionary,
        buidling as many dicts as there are values in the cartesian product of the input sweep dimensions.

        Args:
            sweep_targets (dict[str, Any]): Dictionary containing augmented sweep data containing nodes
            ["path", "values", "partner_path", "partner"], with the latter two being `None` if there are no
            coupled-sweep values.

        Returns:
            list[dict[str, Any]]: list of config dictionaries constructed.
        """
        sweep_keys = [k for k in sweep_targets.keys()]
        lookup_keys = []

        # make a set of lookup keys so we later know where to put the values
        # in the cartesian product.
        for v in sweep_targets.values():
            lookup_keys.append(v["path"])
            if "partner_path" in v:
                for p in v["partner_path"]:
                    lookup_keys.append(p)

        # TODO: look into getting rid of this step. There's optimization potential here,
        # but it works for now.
        # make a list of tuples containing each sweep and partner dimension
        lists = [
            (
                sweep_targets[k]["values"],
                sweep_targets[k]["partner"] if "partner" in sweep_targets[k] else None,
            )
            for k in sweep_keys
        ]

        # make the elements of the cartesian product. each will correspond to one lookup key
        i = 0  # index into `lists`
        elements = []  # container for produced elements
        # have cartesian product done
        self._construct_cartesian_product(
            elements, lists[i][0], lists[i][1], lists, i=i
        )

        # make as many configs as we have values in the cartesian product
        configs = [copy.deepcopy(self.config) for _ in elements]

        # assign the values into the dictionary
        for c, e in zip(configs, elements):
            for k, v in enumerate(e):
                utils.assign_at_path(c, lookup_keys[k], v)

        return configs
