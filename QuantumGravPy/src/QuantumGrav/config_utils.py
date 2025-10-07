from typing import Any
import yaml


def sweep_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> list[Any]:
    print("node: ", node)
    values = node["values"]
    return loader.construct_sequence(values)


def coupled_sweep_constructor(
    loader: yaml.SafeLoader, node: yaml.nodes.MappingNode
) -> list[Any]:
    print("node: ", node)
    values = node["values"]
    data = loader.construct_sequence(values)
    target = loader.construct_sequence(node["target"])
    return list(zip(data, target))


def get_loader():
    loader = yaml.SafeLoader
    loader.add_constructor("!sweep", sweep_constructor)
    loader.add_constructor("!coupled-sweep", coupled_sweep_constructor)
    return loader


cfg = yaml.load(open("./config.yaml"), Loader=get_loader())
