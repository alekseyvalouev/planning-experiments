from __future__ import annotations

import json


class Graph:
    @classmethod
    def deserialize(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        instance = cls.__new__(cls)
        instance.scenes = data["scenes"]
        instance.sparsification_steps = data["sparsification_steps"]
        instance.drop_modality_p = data["drop_modality_p"]
        instance.structured_data = [n["info"] for n in data["nodes"]]
        instance.nodes = [Node(n["node_id"], n["info"]) for n in data["nodes"]]
        node_by_id = {node.node_id: node for node in instance.nodes}
        for n, node in zip(data["nodes"], instance.nodes):
            node.subnodes = n.get("subnodes", node.subnodes)
            for conn in n["connections"]:
                node.add_connection(node_by_id[conn["other_node_id"]])
        return instance


class Node:
    def __init__(self, node_id, info):
        self.node_id = node_id
        self.info = info
        self.modalities = []
        self.subnodes = {}
        self.cache = {}
        if "image" in info:
            self.modalities.append("V")
            self.subnodes["V"] = {"image": info["image"]}
        if "landmarks" in info:
            self.modalities.append("L")
            self.subnodes["L"] = {"landmarks": info["landmarks"]}
        if "image" in info and "landmarks" in info:
            self.modalities.append("VL")
            self.subnodes["VL"] = {
                "image": info["image"],
                "landmarks": info["landmarks"],
            }
        self.connections = []

    def add_connection(self, other_node):
        self.connections.append(other_node)
