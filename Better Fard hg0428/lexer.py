from data import TokenTypes, Operators, Keywords


class Token:
    def __init__(self, toktype, start, end, value):
        if type(toktype) == str:
            toktype = TokenTypes[toktype]
        self.type = toktype
        self.length = end - start
        self.start = start
        self.end = end
        self.value = value

    def __repr__(self):
        return f"Token {self.type.name} at ({self.start}, {self.end}): {self.value}\n"


class Lexer:
    def __init__(self, singleline):
        self.comment = singleline
        self.stringOpen = False
        self.data = ""
        self.index = 0
        self.tokens = []
        self.empty = True
        self.AtEnd = False
        self.curChar = ""

    def isWhitespace(self, char=None):
        char = char or self.curChar
        return char == " " or char == "  " or char == "\t"

    def isNewline(self, char=None):
        char = char or self.curChar
        return char == "\n" or char == ";"

    def isString(self, char=None):
        char = char or self.curChar
        return (char == "\"" or char == "'")

    def isDelimiter(self, char=None):
        char = char or self.curChar
        return (char == ":" or char == "(" or char == ")" or char == ","
                or char == "{" or char == "}" or char == "[" or char == "]"
                or char == ".")

    def detect(self, text):
        if self.curChar == text[0]:
            for i in range(len(text) - 1):
                if not self.peek(i + 1) == text[i + 1]:
                    return False
        else:
            return False
        return True

    def isNumber(self, char=None):
        char = char or self.curChar
        return (char in "0123456789")

    def tokenize(self, data: str):
        if not data: raise ValueError("String cannot be empty")
        self.data += data
        self.curChar = self.data[self.index]
        while self.index < len(self.data):
          
            #Newlines (\n or ;)
            if self.isNewline():
                #Self.empty means that we are on an empty line
                self.empty = True
                self.tokens.append(
                    Token("LineBreak", self.index, self.index, self.curChar))
              
            #Delimiters
            elif self.isDelimiter():
                self.tokens.append(
                    Token("Delimiter", self.index, self.index, self.curChar))
              
            #Numbers
            elif self.isNumber():
                lastIndex = self.index
                value = self.curChar
                while self.isNumber() and not self.AtEnd:
                    self.advance()
                    if not self.isNumber() or self.AtEnd:
                        self.advance(-1)
                        break
                    value += self.curChar
                self.tokens.append(
                    Token("Number", lastIndex, self.index, value))
              
            #Single line comments
            elif self.detect(self.comment): 
                while not self.isNewline() and not self.AtEnd:
                    self.advance()
              
            #Strings
            elif self.isString():
                self.stringOpen = not self.stringOpen
                value = ""
                lastIndex = self.index
                while self.stringOpen and not self.AtEnd:
                    self.advance()
                    if self.isString() or self.AtEnd:
                        self.stringOpen = not self.stringOpen
                        break
                    value += self.curChar

                self.tokens.append(
                    Token("String", lastIndex, self.index, value))
              
            #Directives
            elif self.curChar == "#" and self.empty:
                lastIndex = self.index
                value = ""
                while not self.isWhitespace() and not self.AtEnd:
                    self.advance()
                    if self.AtEnd or self.isWhitespace():
                        break
                    value += self.curChar
                self.tokens.append(
                    Token("Directive", lastIndex, self.index, value))
              
            #Identifiers and Keywords
            elif not (self.isWhitespace() or self.isDelimiter()
                      or self.isNewline() or self.curChar in Operators):
                value = self.curChar
                lastIndex = self.index
                while not (self.isWhitespace() or self.isDelimiter()
                           or self.isNewline()
                           or self.curChar in Operators) and not self.AtEnd:
                    self.advance()
                    if (self.isWhitespace() or self.isDelimiter()
                            or self.isNewline()
                            or self.curChar in Operators) or self.AtEnd:
                        self.advance(-1)
                        break
                    value += self.curChar
                if value in Keywords:
                    self.tokens.append(
                        Token("Keyword", lastIndex, self.index, value))
                else:
                    self.tokens.append(
                        Token("Identifier", lastIndex, self.index, value))
                  
            #Operators
            elif self.curChar in Operators: 
                value = self.curChar
                lastIndex = self.index
                while (not self.AtEnd and value + self.curChar in Operators):
                    self.advance()
                    if self.AtEnd or value + self.curChar not in Operators:
                        self.advance(-1)
                        break
                    value += self.curChar
                self.tokens.append(
                    Token("Operator", lastIndex, self.index, value))
            if self.AtEnd: break
            self.advance()

    def advance(self, amt=1):
        self.index += amt
        if self.index < len(self.data): self.curChar = self.data[self.index]
        else: self.AtEnd = True
        #print(self.curChar)

    def advanceTok(self):
        self.advance(self.tokens[-1].length)

    def peek(self, amt=1):
        if self.index + amt < len(self.data):
            return self.data[self.index + amt]
        else:
            return None