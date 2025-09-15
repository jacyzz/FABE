import os
import json


def root_dir():
    return "/home/nfs/share/backdoor2023/backdoor/attack/IST/transform_v2"


def replace_from_blob(operation, blob):
    diff = 0
    operation = sorted(
        operation,
        key=lambda x: (
            x[0],
            1 if type(x[1]) is int else 0,
            -len(x[1]) if type(x[1]) is not int else 0,
        ),
    )
    for op in operation:
        if type(op[1]) is int:
            if op[1] < 0:
                del_num = op[1]
            else:
                del_num = op[1] - op[0]
            blob = blob[: op[0] + diff + del_num] + blob[op[0] + diff :]
            diff += del_num
        else:
            blob = blob[: op[0] + diff] + op[1] + blob[op[0] + diff :]
            diff += len(op[1])
    return blob


class LanguageNodeTypeMap:
    def __init__(self, lang):
        with open(
            os.path.join(root_dir(), "config", "language_node_type.json"), "r"
        ) as f:
            self.map = json.loads(f.read())
        self.lang = lang

    def __getitem__(self, struct):
        return self.map[struct][self.lang]


class TreeDFSer:
    def __init__(self):
        pass

    def find_valid_nodes(self, node, check_valid_func=None):
        assert check_valid_func is not None
        valid_nodes = []
        if check_valid_func(node):
            valid_nodes.append(node)
        for child in node.children:
            valid_nodes.extend(self.find_valid_nodes(child, check_valid_func))
        return valid_nodes


if __name__ == "__main__":
    lang_node_type_map = LanguageNodeTypeMap()
    print(lang_node_type_map.get("function", "c"))
