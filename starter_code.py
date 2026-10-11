# To run this script, you need to install the Graphviz Python library using pip:
# pip install graphviz

from dataclasses import dataclass, field

import graphviz


@dataclass
class Node:
    node_id: int
    node_type: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    from_node: int
    from_port: int
    to_node: int
    to_port: int
    shape: tuple[int, ...]


NODE_FILLCOLOR_BY_TYPE = {
    "Input": "#EDEADE",  # Alabaster
    "Constant": "#FFD54F",  # Amber
    "MatMul": "#90CAF9",  # Blue
    "Add": "#B39DDB",  # Purple
    "Output": "#EDEADE",  # Alabaster
}


def convert_node_to_label(v: Node) -> str:
    new_line = r"\l"
    rows = [f"{attr_name}:{attr_value}{new_line}" for attr_name, attr_value in v.attributes.items()]
    rows = "".join(rows)
    begin = "{"
    end = "}"
    label = f"{begin}{v.node_type}|node_id:{v.node_id}{new_line}{rows}{end}"
    return label


def main():
    nodes = [
        Node(0, "Input"),
        Node(1, "Constant", {"dtype": "fp32"}),
        Node(2, "MatMul"),
        Node(3, "Constant", {"dtype": "fp32"}),
        Node(4, "Add"),
        Node(5, "Output"),
    ]
    edges = [
        Edge(0, 0, 2, 0, (1, 768)),
        Edge(1, 0, 2, 1, (768, 1024)),
        Edge(2, 0, 4, 0, (1, 1024)),
        Edge(3, 0, 4, 1, (1, 1024)),
        Edge(4, 0, 5, 0, (1, 1024)),
    ]

    dg = graphviz.Digraph(format="svg")

    # Configure graph attributes
    dg.graph_attr["ranksep"] = "0.2"
    # Configure node attributes
    dg.node_attr["style"] = "rounded,filled"
    dg.node_attr["shape"] = "record"
    dg.node_attr["fontsize"] = "10"
    dg.node_attr["fontname"] = "Tahoma"
    dg.node_attr["fillcolor"] = "white"
    # Configure edge attributes
    dg.edge_attr["arrowsize"] = "0.5"
    dg.edge_attr["fontsize"] = "10"
    dg.edge_attr["fontname"] = "Tahoma"

    for v in nodes:
        dg.node(
            str(v.node_id),
            label=convert_node_to_label(v),
            fillcolor=NODE_FILLCOLOR_BY_TYPE[v.node_type],
            id=str(v.node_id),
        )

    for e in edges:
        shape = ",".join(map(str, e.shape))
        dg.edge(str(e.from_node), str(e.to_node), label=f"[{shape}]")

    dg.render("simple_graph", cleanup=True)


if __name__ == "__main__":
    main()
