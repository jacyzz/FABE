from ist_utils import text, print_children
from transform.lang import get_lang, get_expand


def get_indent(start_byte, code):
    indent = 0
    i = start_byte
    try:
        while len(code) > 0 and len(code) < i and i >= 0 and code[i] != "\n":
            if code[i] == " ":
                indent += 1
            elif code[i] == "\t":
                indent += 4
            i -= 1
    except:
        print(f"code = {code}")
        print(f"i = {i}")
        pass
    return indent


def match_ifforwhile_has_bracket(root):
    lang = get_lang()
    block_mp = {"c": "compound_statement", "java": "block", "python": "block"}

    def check(u):
        # 支持的控制流语句类型
        control_statements = ["while_statement", "if_statement", "for_statement", "else_clause"]
        if get_lang() == "python":
            # Python使用缩进而不是大括号，需要不同的检测逻辑
            if u.type in control_statements:
                # 对于Python，检查是否只有单个语句
                pass  # 继续执行下面的检查逻辑
        else:
            # C/Java使用大括号
            if not (u.type in control_statements and "{" in text(u) and "}" in text(u)):
                return False
        
        if u.type in control_statements:
            count = -1
            for v in u.children:
                if v.type == block_mp[lang]:
                    count = 0
                    for p in v.children:
                        # 支持的语句类型（C, Java, Python）
                        supported_statements = [
                            "expression_statement",
                            "return_statement", 
                            "compound_statement",
                            "break_statement",
                            "for_statement",
                            "if_statement", 
                            "while_statement",
                            # Python特有语句类型
                            "assignment",
                            "print_statement",
                            "continue_statement",
                        ]
                        if p.type in supported_statements:
                            count += 1
            if -1 < count <= 1:
                return True
        return False

    res = []

    def match(u):
        if check(u):
            res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def match_ifforwhile_hasnt_bracket(root):
    res = []

    def match(u):
        if not get_expand():
            control_statements = ["while_statement", "if_statement", "for_statement", "else_clause"]
            # 统一计算匹配条件，避免在不匹配时提前return导致跳过子树
            if get_lang() == "python":
                cond = u.type in control_statements
            else:
                # C/Java: 无大括号
                cond = (u.type in control_statements and "{" not in text(u) and "}" not in text(u))

            if cond:
                res.append(u)

        elif get_expand():
            if (
                u.type
                in [
                    "while_statement",
                    "if_statement",
                    "for_statement",
                    "else_clause",
                    "return_statement",
                    "expression_statement",
                    "throw_statement",
                ]
                and text(u)[0] != "{"
            ):
                # There is only one 'expression_statement'
                if u.type == "expression_statement":
                    if "expression_statement" in [t.type for t in res]:
                        return
                res.append(u)
        for v in u.children:
            match(v)

    match(root)
    return res


def convert_del_ifforwhile_bracket(node, code):
    # Remove braces in single line If, For, While
    lang = get_lang()
    block_mp = {"c": "compound_statement", "java": "block", "python": "block"}
    statement_node = None
    # Python: 将多行缩进块折叠为单行语句
    if lang == "python":
        contents = text(node)
        if "\n" not in contents:
            return
        # 形如: "if cond:\n    stmt" -> "if cond: stmt"
        lines = contents.splitlines()
        if len(lines) < 2:
            return
        head = lines[0]
        # 找到第一条非空语句行
        stmt_line = ""
        for ln in lines[1:]:
            if ln.strip():
                stmt_line = ln.strip()
                break
        if not stmt_line:
            return
        new_contents = f"{head} {stmt_line}"
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, new_contents),
        ]

    for u in node.children:
        if u.type == block_mp[lang]:
            statement_node = u
            break
    if statement_node is None:
        return
    contents = text(statement_node)
    new_contents = contents.replace("{", "").replace("}", "").replace("\n", "")
    tmps = []
    for new_content in new_contents.split(";"):
        for i in range(len(new_content)):
            if new_content[i] != " ":
                tmps.append(" " + new_content[i:])
                break
    indent = get_indent(node.start_byte, code)
    new_contents = ", ".join(tmps) + ";\n" + " " * indent

    return [
        (statement_node.end_byte, statement_node.start_byte),
        (statement_node.start_byte, new_contents),
    ]


def convert_add_ifforwhile_bracket(node, code):
    if node.type in ["return_statement", "expression_statement", "throw_statement"]:
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, f"{{{text(node)}}}"),
        ]
    # Add braces to single line If, For, While
    statement_node = None
    if get_lang() == "python":
        # Python: 将单行 if/for/while 扩展为多行缩进块
        contents = text(node)
        if "\n" in contents:
            return
        # 按第一个冒号分割
        if ":" not in contents:
            return
        head, tail = contents.split(":", 1)
        tail_stmt = tail.strip()
        indent = get_indent(node.start_byte, code)
        new_contents = f"{head}:\n{(indent + 4) * ' '}{tail_stmt}"
        return [
            (node.end_byte, node.start_byte),
            (node.start_byte, new_contents),
        ]
    for each in node.children:
        if each.type in [
            "expression_statement",
            "return_statement",
            "compound_statement",
            "break_statement",
            "for_statement",
            "if_statement",
            "while_statement",
        ]:
            statement_node = each
    if statement_node is None:
        return
    indent = get_indent(node.start_byte, code)

    if statement_node.prev_sibling is None:
        return
    if "\n" not in text(node):
        return [
            (statement_node.start_byte, statement_node.prev_sibling.end_byte),
            (statement_node.start_byte, f" {{\n{(indent + 4) * ' '}"),
            (statement_node.end_byte, f"\n{indent * ' '}}}"),
        ]
    else:
        return [
            (statement_node.prev_sibling.end_byte, f" {{"),
            (statement_node.end_byte, f"\n{indent * ' '}}}"),
        ]


def count_has_ifforwhile_bracket(root):
    if get_expand():
        lang = get_lang()
        block_mp = {"c": "compound_statement", "java": "block", "python": "block"}

        def check(u):
            if u.type == block_mp[lang] and u.parent.type != "method_declaration":
                return True
            return False

        res = []

        def match(u):
            if check(u):
                res.append(u)
            for v in u.children:
                match(v)

        match(root)
        return len(res)

    nodes = match_ifforwhile_has_bracket(root)
    return len(nodes)


def count_hasnt_ifforwhile_bracket(root):
    nodes = match_ifforwhile_hasnt_bracket(root)
    return len(nodes)
