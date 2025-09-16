import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import text
from transform.lang import get_lang
import random

record = {}


def match_tokensub_identifier(root, select=True):
    parameter_declaration_sons = {}

    lang = get_lang()

    if lang == "python":
        # Python: 仅选择函数形参标识符，避免重命名函数、属性、模块名等
        def check(node):
            if node.type != "identifier":
                return False
            p = node.parent
            # 排除函数/类名、调用、属性、导入
            if p.type in [
                "function_definition",
                "class_definition",
                "call",
                "attribute",
                "import_statement",
                "import_from_statement",
                "aliased_import",
            ]:
                return False
            # 收集形参
            u = node
            while u is not None:
                if u.type == "parameters":
                    parameter_declaration_sons[text(node)] = True
                    break
                u = u.parent
            return True
    else:
        def check(node):
            if node.type == "identifier":
                if node.parent.type == "function_declarator":
                    return False
                if node.parent.type == "call_expression":
                    return False
                if node.parent.type == "parameter_declaration":
                    parameter_declaration_sons[text(node)] = True
                return True
            return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    res = [node for node in res if text(node) in parameter_declaration_sons]
    if select:
        res = [node for node in res if len(text(node)) > 0]
        if len(res) == 0:
            return res
        selected_var_name = random.choice([text(node) for node in res])
        res = [
            node
            for node in res
            if len(text(node)) > 0 and text(node) == selected_var_name
        ]
        record["insert_position"] = random.choice(["suffix", "prefix"])
    return res


def convert_tokensub_rb(node):
    if record["insert_position"] == "suffix":
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, "_".join([text(node), "rb"])),
        ]
    else:
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, "_".join(["rb", text(node)])),
        ]


def convert_tokensub_sh(node):
    if record["insert_position"] == "suffix":
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, "_".join([text(node), "sh"])),
        ]
    else:
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, "_".join(["sh", text(node)])),
        ]


def count_tokensub_rb(root):
    for node in match_tokensub_identifier(root):
        if len(text(node).split("_")) > 1:
            if "rb" in text(node).split("_"):
                return 1
    return 0


def count_tokensub_sh(root):
    count = 0
    for node in match_tokensub_identifier(root, select=False):
        if len(text(node).split("_")) > 1:
            if "sh" in text(node).split("_"):
                count += 1
    return count
