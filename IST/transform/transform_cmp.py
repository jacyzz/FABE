from ist_utils import text, print_children
from transform.lang import get_lang
import re


def match_cmp(root):
    # a ? b
    res = []

    def check(u):
        # C/Java: binary_expression
        if u.type == "binary_expression" and len(u.children) == 3 and text(u.children[1]) in [">", ">=", "<", "<=", "==", "!="]:
            return True
        # Python: comparison 节点，且仅包含一个比较运算符
        if get_lang() == "python" and u.type == "comparison":
            s = text(u)
            # 简化：仅匹配 a OP b 的简单形式
            m = re.match(r"^\s*[^\n:]+?(==|!=|<=|>=|<|>)\s*[^\n:]+?$", s)
            return m is not None
        return False

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def match_bigger(root):
    # a >/>= b
    res = []

    def check(u):
        if u.type == "binary_expression" and text(u.children[1]) in [">", ">="]:
            return True
        if get_lang() == "python" and u.type == "comparison":
            s = text(u)
            return re.search(r"(>|>=)", s) is not None and re.search(r"(<=|<|==|!=)", s) is None
        return False

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def match_smaller(root):
    # a </<= b
    res = []

    def check(u):
        if u.type == "binary_expression" and text(u.children[1]) in ["<", "<="]:
            return True
        if get_lang() == "python" and u.type == "comparison":
            s = text(u)
            return re.search(r"(<|<=)", s) is not None and re.search(r"(>=|>|==|!=)", s) is None
        return False

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def match_equal(root):
    # a </<= b
    res = []

    def check(u):
        if u.type == "binary_expression" and text(u.children[1]) in ["=="]:
            return True
        if get_lang() == "python" and u.type == "comparison":
            s = text(u)
            # 仅包含 ==
            return ("==" in s) and not any(op in s for op in ["!=","<=","<",">=",">"])
        return False

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def match_not_equal(root):
    # a </<= b
    res = []

    def check(u):
        if u.type == "binary_expression" and text(u.children[1]) in ["!="]:
            return True
        if get_lang() == "python" and u.type == "comparison":
            s = text(u)
            # 仅包含 !=
            return ("!=" in s) and not any(op in s for op in ["==","<=","<",">=",">"])
        return False

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)

    return res


def convert_smaller(node):
    if get_lang() == "python":
        s = text(node)
        # 将 a >= b / a > b 反转为 b <= a / b < a；等价逻辑与下方一致
        if ">=" in s:
            a, b = [p.strip() for p in re.split(r">=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{b} <= {a}")]
        if ">" in s:
            a, b = [p.strip() for p in re.split(r">", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{b} < {a}")]
        if "==" in s:
            a, b = [p.strip() for p in re.split(r"==", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} <= {b} and {b} <= {a}")]
        if "!=" in s:
            a, b = [p.strip() for p in re.split(r"!=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} < {b} or {b} < {a}")]
    [a, op, b] = [text(x) for x in node.children]
    if op in ["<=", "<"]:
        return
    if op in [">=", ">"]:
        reverse_op_dict = {">": "<", ">=": "<="}
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{b} {reverse_op_dict[op]} {a}"),
        ]
    if op in ["=="]:
        # b <= a && a <= b
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} <= {b} && {b} <= {a}"),
        ]
    if op in ["!="]:
        # a < b || b < a
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} < {b} || {b} < {a}"),
        ]


def convert_bigger(node):
    if get_lang() == "python":
        s = text(node)
        if "<=" in s:
            a, b = [p.strip() for p in re.split(r"<=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{b} >= {a}")]
        if "<" in s:
            a, b = [p.strip() for p in re.split(r"<", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{b} > {a}")]
        if "==" in s:
            a, b = [p.strip() for p in re.split(r"==", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} >= {b} and {b} >= {a}")]
        if "!=" in s:
            a, b = [p.strip() for p in re.split(r"!=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} > {b} or {b} > {a}")]
    [a, op, b] = [text(x) for x in node.children]
    if op in [">=", ">"]:
        return
    if op in ["<=", "<"]:
        reverse_op_dict = {"<": ">", "<=": ">="}
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{b} {reverse_op_dict[op]} {a}"),
        ]
    if op in ["=="]:
        # a >= b && b >= a
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} >= {b} && {b} >= {a}"),
        ]
    if op in ["!="]:
        # a > b || b > a
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} > {b} || {b} > {a}"),
        ]


def convert_equal(node):
    if get_lang() == "python":
        s = text(node)
        if "<=" in s:
            a, b = [p.strip() for p in re.split(r"<=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} < {b} or {a} == {b}")]
        if "<" in s:
            a, b = [p.strip() for p in re.split(r"<", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"not ({b} < {a} or {a} == {b})")]
        if ">=" in s:
            a, b = [p.strip() for p in re.split(r">=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} > {b} or {a} == {b}")]
        if ">" in s:
            a, b = [p.strip() for p in re.split(r">", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"not ({b} > {a} or {a} == {b})")]
        if "!=" in s:
            a, b = [p.strip() for p in re.split(r"!=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"not ({a} == {b})")]
    [a, op, b] = [text(x) for x in node.children]
    if op in ["=="]:
        return
    if op in ["<="]:
        # a < b || a == b
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} < {b} || {a} == {b}"),
        ]
    if op in ["<"]:
        # !(b < a || a == b)
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"!({b} < {a} || {a} == {b})"),
        ]
    if op in [">="]:
        # a > b || a == b
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} > {b} || {a} == {b}"),
        ]
    if op in [">"]:
        # !(b > a || a == b)
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"!({b} > {a} || {a} == {b})"),
        ]
    if op in ["!="]:
        # !(a == b)
        return [(node.end_byte, node.start_byte), (node.start_byte, f"!({a} == {b})")]


def convert_not_equal(node):
    if get_lang() == "python":
        s = text(node)
        if "<=" in s:
            a, b = [p.strip() for p in re.split(r"<=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"not ({b} < {a} and {a} != {b})")]
        if "<" in s:
            a, b = [p.strip() for p in re.split(r"<", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} < {b} and {a} != {b}")]
        if ">=" in s:
            a, b = [p.strip() for p in re.split(r">=", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"not ({b} > {a} and {a} != {b})")]
        if ">" in s:
            a, b = [p.strip() for p in re.split(r">", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"{a} < {b} and {a} != {b}")]
        if "==" in s:
            a, b = [p.strip() for p in re.split(r"==", s, maxsplit=1)]
            return [(node.end_byte, node.start_byte), (node.start_byte, f"not ({a} != {b})")]
    [a, op, b] = [text(x) for x in node.children]
    if op in ["!="]:
        return
    if op in ["<="]:
        # !(b < a && a != b)
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"!({b} < {a} && {a} != {b})"),
        ]
    if op in ["<"]:
        # (a < b && a != b)
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} < {b} && {a} != {b}"),
        ]
    if op in [">="]:
        # !(b > a && a != b)
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"!({b} > {a} && {a} != {b})"),
        ]
    if op in [">"]:
        # a < b && a != b
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{a} < {b} && {a} != {b}"),
        ]
    if op in ["=="]:
        # !(a != b)
        return [(node.end_byte, node.start_byte), (node.start_byte, f"!({a} != {b})")]


def count_bigger(root):
    nodes = match_bigger(root)
    return len(nodes)


def count_smaller(root):
    nodes = match_smaller(root)
    return len(nodes)


def count_equal(root):
    nodes = match_equal(root)
    return len(nodes)


def count_not_equal(root):
    nodes = match_not_equal(root)
    return len(nodes)
