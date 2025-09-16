from ist_utils import text, print_children
from transform.lang import get_lang
import re

return_text = None


def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    while i >= 0 and code[i] != "\n":
        if code[i] == " ":
            indent += 1
        elif code[i] == "\t":
            indent += 8
        i -= 1
    return indent


"""==========================match========================"""


def match_if_equivalent(root):
    def check(node):
        if node.type != "if_statement":
            return False
        # Python 分支：使用文本快速判断条件可被等价转换
        if get_lang() == "python":
            s = text(node)
            # 形如: if <cond>:
            if not s.strip().startswith("if") or ":" not in s:
                return False
            cond = s.strip()[2:s.index(":")].strip()
            # 简化规则：包含比较运算符或存在 not
            return any(op in cond for op in ["==", "!=", "<=", ">=", "<", ">"]) or cond.startswith("not ")
        # C/Java 分支（保持原逻辑）
        expr_node = node.children[1]
        expr_in_node = expr_node.children[1]
        while len(expr_in_node.children) >= 2 and text(expr_in_node.children[0]) == "(":
            expr_in_node = expr_in_node.children[1]
        if len(expr_in_node.children) < 3:
            return False
        return True

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


"""=========================replace========================"""


def cvt_equivalent(node, code):
    # Python 分支：直接基于文本进行等价转换
    if get_lang() == "python":
        s = text(node)
        # 拆分:  if <cond>:<rest>
        try:
            head, tail = s.split(":", 1)
        except ValueError:
            return
        prefix = head[: head.find("if")]
        cond = head[head.find("if") + 2 :].strip()
        rest = ":" + tail
        # 规则：
        # - 若含比较运算符，翻转比较运算符（与 C/Java 一致）
        # - 若以 not 开头，去掉 not
        # - 否则加 not 前缀
        def flip_compare(c: str) -> str:
            # 只处理最简单 a OP b 形式
            for op, op2 in [("==","!="),("!=","=="),(">","<="),("<=", ">"),("<", ">="),(">=", "<")]:
                if op in c:
                    parts = c.split(op, 1)
                    return f"{parts[0].strip()} {op2} {parts[1].strip()}"
            return None

        flipped = flip_compare(cond)
        if flipped is not None:
            new_cond = flipped
        elif cond.startswith("not "):
            new_cond = cond[4:].strip()
        else:
            new_cond = f"not {cond}"
        new_if = f"{prefix}if {new_cond}{rest}"
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, new_if),
        ]

    # C/Java 分支：保持原有 AST 驱动逻辑
    if_node = node.children[0]
    expr_node = node.children[1]
    expr_in_node = expr_node.children[1]
    new_str = ""
    opposite = {
        "==": "!=",
        "!=": "==",
        ">": "<=",
        "<=": ">",
        "<": ">=",
        ">=": "<",
        "&&": "||",
        "||": "&&",
    }
    mp = {}

    def opp_dfs(u):
        while len(u.children) >= 2 and text(u.children[0]) == "(":
            u = u.children[1]
        if len(u.children) == 0:
            mp[text(u)] = "!" + text(u)
            return
        if "&&" not in text(u) and "||" not in text(u) and len(u.children) == 3:
            if text(u.children[1]) in opposite:
                mp[text(u)] = (
                    text(u.children[0])
                    + opposite[text(u.children[1])]
                    + text(u.children[2])
                )
            return
        elif len(u.children) == 2 and text(u.children[0]) == "!":
            mp[text(u)] = text(u.children[1])
            return
        elif len(u.children) == 3:
            try:
                mp[text(u)] = (
                    text(u.children[0])
                    + opposite[text(u.children[1])]
                    + text(u.children[2])
                )
            except:
                mp[text(u)] = "!" + text(u)

        for v in u.children:
            opp_dfs(v)

    opp_dfs(expr_in_node)
    new_str = text(expr_in_node)
    for key, val in mp.items():
        new_str = new_str.replace(key, val)
    new_str = "!(" + new_str + ")"
    return [
        (expr_in_node.end_byte, expr_in_node.start_byte),
        (expr_in_node.start_byte, new_str),
    ]
