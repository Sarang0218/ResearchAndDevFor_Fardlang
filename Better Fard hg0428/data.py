TokenTypes = {}


class TokenType:
    def __init__(self, name):
        self.name = name
        TokenTypes[name] = self


TokenType("String")
TokenType("Number")
TokenType("LineBreak")
TokenType("Directive")
TokenType("Keyword")
TokenType("Operator")
TokenType("Identifier")
TokenType("Delimiter")

AssignmentOps = [
    "=", "+=", "-=", "*=", "/=", "//=", "^=", "%=", "|=", "++", "--"
]  # These operators assign values

Operators = AssignmentOps + [
    "&",
    "?"
    "=",
    "+",
    "-",
    "/",
    "*",
    "^",
    "%",
    "!",
    "<",
    ">",
    "==",
    "<=",
    ">=",
    "!=",
    "&&",
    "||",
    "x|",
]  # These operators are just operators

Keywords = ["refard", "farding", "fards?", "fard", "farded"]
