from data import TokenTypes, Keywords
from lexer import Token


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.lines = self.generateLines()
        self.AST = {"Directives": {}, "Code": []}
        self.line = 0
        self.column = 0
        self.index = 0
        self.toknum = 0
        self.curTok = None

    def Type(self, t, tok=None):
        tok = tok or self.curTok
        return tok.type == TokenTypes[t]

    def generateLines(self):
        lines = [[]]
        lnum = 0
        for tok in self.tokens:
            if self.Type("LineBreak", tok):
                lines.append([])
                lnum += 1
                continue
            lines[lnum].append(tok)
        return lines

    def advance(self, amt=1):
        self.index += amt
        if self.index < len(self.tokens):
            self.column += self.curTok.end - self.curTok.start
            self.curTok = self.tokens[self.index]
            self.toknum += 1
            if self.Type('LineBreak'):
                self.line += 1
                self.toknum = -1
                self.column = 0
        else:
            self.AtEnd = True

    def advanceLine(self, amt=1):
        for i in self.lines[self.line:self.line + amt]:
            self.advance(len(i))

    def peek(self, amt=1):
        if self.index + amt < len(self.tokens):
            return self.tokens[self.index + amt]
        else:
            return None
