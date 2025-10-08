from ist_utils import text, print_children
from transform.lang import get_lang
import re


def _in_condition_context(node):
    """Only treat comparisons inside boolean conditions as matches (C/Java).
    - if/while/do (...)
    - for (init; cond; update)
    Guard against expressions in statement bodies by stopping at block nodes.
    """
    cur = node
    # Walk up but stop if we hit a block/body container first
    while cur is not None:
        parent = getattr(cur, 'parent', None)
        if parent is None:
            return False
        # If we reach a block/body before control header, it's not in condition
        if parent.type in ("block", "compound_statement"):
            return False
        # Parenthesized condition of if/while/do
        if parent.type == "parenthesized_expression":
            gp = getattr(parent, 'parent', None)
            if gp and gp.type in ("if_statement", "while_statement", "do_statement"):
                return True
        # For header: allow expressions that ascend to for_statement before hitting a block
        if parent.type == "for_statement":
            return True
        cur = parent
    return False


def match_cmp(root):
    # a ? b
    res = []

    def check(u):
        # C/Java: binary_expression strictly in condition context
        if (
            u.type == "binary_expression"
            and len(u.children) == 3
            and text(u.children[1]) in [">", ">=", "<", "<=", "==", "!="]
            and get_lang() in ("c", "java", "c_sharp")
            and _in_condition_context(u)
        ):
            return True
        # Python: 在 if_statement 的 condition 上判断
        if get_lang() == "python" and u.type == "if_statement":
            cond = u.child_by_field_name("condition")
            if cond is None:
                return False
            s = text(cond)
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
        if (
            u.type == "binary_expression"
            and text(u.children[1]) in [">", ">="]
            and get_lang() in ("c", "java", "c_sharp")
            and _in_condition_context(u)
        ):
            return True
        if get_lang() == "python" and u.type == "if_statement":
            cond = u.child_by_field_name("condition")
            if cond is None:
                return False
            s = text(cond)
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
        if (
            u.type == "binary_expression"
            and text(u.children[1]) in ["<", "<="]
            and get_lang() in ("c", "java", "c_sharp")
            and _in_condition_context(u)
        ):
            return True
        if get_lang() == "python" and u.type == "if_statement":
            cond = u.child_by_field_name("condition")
            if cond is None:
                return False
            s = text(cond)
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
        if (
            u.type == "binary_expression"
            and text(u.children[1]) in ["=="]
            and get_lang() in ("c", "java", "c_sharp")
            and _in_condition_context(u)
        ):
            return True
        if get_lang() == "python" and u.type == "if_statement":
            cond = u.child_by_field_name("condition")
            if cond is None:
                return False
            s = text(cond)
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
        if (
            u.type == "binary_expression"
            and text(u.children[1]) in ["!="]
            and get_lang() in ("c", "java", "c_sharp")
            and _in_condition_context(u)
        ):
            return True
        if get_lang() == "python" and u.type == "if_statement":
            cond = u.child_by_field_name("condition")
            if cond is None:
                return False
            s = text(cond)
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
        cond_node = node.child_by_field_name("condition") if node.type == "if_statement" else node
        s = text(cond_node)
        # 将 a >= b / a > b 反转为 b <= a / b < a；等价逻辑与下方一致
        if ">=" in s:
            a, b = [p.strip() for p in re.split(r">=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{b} <= {a}")]
        if ">" in s:
            a, b = [p.strip() for p in re.split(r">", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{b} < {a}")]
        if "==" in s:
            a, b = [p.strip() for p in re.split(r"==", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} <= {b} and {b} <= {a}")]
        if "!=" in s:
            a, b = [p.strip() for p in re.split(r"!=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} < {b} or {b} < {a}")]
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
            (node.start_byte, f"({a} <= {b} && {b} <= {a})"),
        ]
    if op in ["!="]:
        # a < b || b < a
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"({a} < {b} || {b} < {a})"),
        ]


def convert_bigger(node):
    if get_lang() == "python":
        cond_node = node.child_by_field_name("condition") if node.type == "if_statement" else node
        s = text(cond_node)
        if "<=" in s:
            a, b = [p.strip() for p in re.split(r"<=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{b} >= {a}")]
        if "<" in s:
            a, b = [p.strip() for p in re.split(r"<", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{b} > {a}")]
        if "==" in s:
            a, b = [p.strip() for p in re.split(r"==", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} >= {b} and {b} >= {a}")]
        if "!=" in s:
            a, b = [p.strip() for p in re.split(r"!=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} > {b} or {b} > {a}")]
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
        cond_node = node.child_by_field_name("condition") if node.type == "if_statement" else node
        s = text(cond_node)
        if "<=" in s:
            a, b = [p.strip() for p in re.split(r"<=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} < {b} or {a} == {b}")]
        if "<" in s:
            a, b = [p.strip() for p in re.split(r"<", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"not ({b} < {a} or {a} == {b})")]
        if ">=" in s:
            a, b = [p.strip() for p in re.split(r">=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} > {b} or {a} == {b}")]
        if ">" in s:
            a, b = [p.strip() for p in re.split(r">", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"not ({b} > {a} or {a} == {b})")]
        if "!=" in s:
            a, b = [p.strip() for p in re.split(r"!=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"not ({a} == {b})")]
    [a, op, b] = [text(x) for x in node.children]
    if op in ["=="]:
        return
    if op in ["<="]:
        # a < b || a == b
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"({a} < {b} || {a} == {b})"),
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
            (node.start_byte, f"({a} > {b} || {a} == {b})"),
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
        cond_node = node.child_by_field_name("condition") if node.type == "if_statement" else node
        s = text(cond_node)
        if "<=" in s:
            a, b = [p.strip() for p in re.split(r"<=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"not ({b} < {a} and {a} != {b})")]
        if "<" in s:
            a, b = [p.strip() for p in re.split(r"<", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} < {b} and {a} != {b}")]
        if ">=" in s:
            a, b = [p.strip() for p in re.split(r">=", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"not ({b} > {a} and {a} != {b})")]
        if ">" in s:
            a, b = [p.strip() for p in re.split(r">", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"{a} < {b} and {a} != {b}")]
        if "==" in s:
            a, b = [p.strip() for p in re.split(r"==", s, maxsplit=1)]
            return [(cond_node.end_byte, cond_node.start_byte), (cond_node.start_byte, f"not ({a} != {b})")]
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
            (node.start_byte, f"({a} < {b} && {a} != {b})"),
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
            (node.start_byte, f"({a} < {b} && {a} != {b})"),
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
