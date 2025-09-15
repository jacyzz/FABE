import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ist_utils import get_indent, text, print_children
from transform_v2.utils import LanguageNodeTypeMap, TreeDFSer, replace_from_blob

from tree_sitter import Parser, Language


class DeadCode:
    def __init__(self, language, transform_type):
        self.language = language
        self.transform_type = transform_type

        self.lnt_map = LanguageNodeTypeMap()
        self.tree_dfser = TreeDFSer()

        self.deadcode_map = {
            "c": {
                "if-testmsg": 'if(1 == -1){ printf("INFO Test message:aaaaa");}',
                "if-233": 'if(1 == -1){ printf("233");}',
            },
            "java": {
                "if-testmsg": 'if(1 == -1){ System.out.println("INFO Test message:aaaaa");}',
                "if-233": 'if(1 == -1){ System.out.println("233");}',
            },
            "c_sharp": {
                "if-testmsg": 'if(1 == -1){ Console.WriteLine("INFO Test message:aaaaa");}',
                "if-233": 'if(1 == -1){ Console.WriteLine("233");}',
            },
            "python": {
                "if-testmsg": 'if 1 == -1: print("INFO Test message:aaaaa")',
                "if-233": 'if 1 == -1: print("233")',
            },
        }

    def _check_match_valid(self, node):
        # only match function
        return node.type == self.lnt_map["function"]

    def _count(self, root):
        if self.transform_type == "if-testmsg":
            return "INFO Test message:aaaaa" in text(root)
        elif self.transform_type == "if-233":
            return "233" in text(root)

    def match(self, root):
        return self.tree_dfser.find_valid_nodes(root, self._check_match_valid)

    def convert(self, node, code):
        block_node = None
        for child in node.children:
            if child.type == self.lnt_map["block"]:
                block_node = child
                break
        if block_node is None:
            return

        deadcode = self.deadcode_map[self.language][self.transform_type]
        indent = get_indent(block_node.children[1].start_byte, code)

        return [(block_node.children[0].end_byte, f"\n{' '*indent}{deadcode}")]

    def count(self, root):
        return sum([self._count(node) for node in self.match(root)])

    def tranform(self, code):
        parent_dir = os.path.dirname(__file__)
        languages_so_path = os.path.join(
            parent_dir, "build", f"{self.language}-languages.so"
        )
        parser = Parser()
        parser.set_language(Language(languages_so_path, self.language))
        self.parser = parser

        AST = self.parser.parse(bytes(code, encoding="utf-8"))
        valid_nodes = self.match(AST.root_node)

        if len(valid_nodes) == 0:
            return code, False

        operations = []
        for node in valid_nodes:
            operations.extend(self.convert(node, code))

        transformed_code = replace_from_blob(operations, code)
        succ = code.replace(" ", "").replace("\n", "").replace(
            "\t", ""
        ) != transformed_code.replace(" ", "").replace("\n", "").replace("\t", "")

        return transformed_code, succ
