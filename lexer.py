class Token:
	def __init__(self, token, typef="identifer", start=0, end=0):
		self.value = token
		self.start = start
		self.end = end
		self.type = typef
	
	def equals(self, tokOrType, value=""):
		if type(tokOrType) == Token:
			return tokOrType.type == self.type and tokOrType.value == self.value
		return tokOrType == self.type and value == self.value
	
	def not_equals(self, tokOrType, value=""):
		return not self.equals(tokOrType, value)
		
	def __repr__(self):
		return f"Token({self.type}, {self.value}, from {self.start} to {self.end})"
		return f"\n-----------\nTOKEN (\n\tType:{self.type},\n\tValue:{self.value},\n\tfrom {self.start} to {self.end}\n)\n-----------"

from error import Error

import string
class Lexer:
	def __init__(self, code):
		self.code = code
		self.ind = 0
		self.cur = self.code[self.ind] if self.ind < len(self.code) else None
		self.DIGITS = "01234567890"
		self.LETTERS = string.ascii_letters + "_"
		self.SPECIAL_KEYWORD_LETTERS = "?"

		self.keywords = [
			"refard",
			"farding",
			"fards?",
      "fard",
      "farded"
		]
		self.other = [
      {
        "name": "BOOL",
        "match": "true"
      },
      {
        "name": "BOOL",
        "match": "false"
      },
      {
        "name": "LOGICAL_OP",
        "match": "=="
      },
      {
        "name": "LOGICAL_OP",
        "match": ">="
      },
      {
        "name": "LOGICAL_OP",
        "match": "<="
      },{
        "name": "LOGICAL_OP",
        "match": ">"
      },
      {
        "name": "LOGICAL_OP",
        "match": "<"
      },
      {
        "name": "LOGICAL_OP",
        "match": "!="
      },
      {
        "name": "SPECIAL",
        "match": "&"
      },
			{
				"name": "VARIABLE_MODIFY",
				"match": "+="
			},
			{
				"name": "VARIABLE_MODIFY",
				"match": "-="
			},
			{
				"name": "VARIABLE_MODIFY",
				"match": "*="
			},
			{
				"name": "VARIABLE_MODIFY",
				"match": "/="
			},
			{
				"name": "PLUS",
				"match": "+"
			},
			{
				"name": "MINUS",
				"match": "-"
			},
			{
				"name": "TIMES",
				"match": "*"
			},
			{
				"name": "DIV",
				"match": "/"
			},
			{
				"name": "LPAREN",
				"match": "("
			},
			{
				"name": "RPAREN",
				"match": ")"
			},
			{
				"name": "EQUALS",
				"match": "="
			}
		]
		
	def advance(self):
		self.ind += 1
		self.cur = self.code[self.ind] if self.ind < len(self.code) else None
		
	def back(self):
		self.ind -= 1
		self.cur = self.code[self.ind] if self.ind >= 0 else None

	def tokenizeFard(self):
		toks = []
		while self.cur:
			broke = False
			for rule in self.other:
				if rule["match"].startswith(self.cur):
					text = ""
					start = self.ind
					for _ in range(len(rule["match"])):
						text += self.code[self.ind + _]
					if text == rule["match"]:
						toks.append(Token(text, rule["name"], start, self.ind + len(rule["match"]) - 1))
						for _ in range(len(rule["match"]) - 1): self.advance()
						broke = True
						break
			
			if broke:
				pass

			elif self.cur == '"':
				strng = ""
				start = self.ind
				self.advance()
				while self.cur:
					if self.cur == '"':
						if len(strng) > 0 and strng[-1] == "\\":
							pass
						else:
							break
					strng += self.cur
					self.advance()
				toks.append(Token(strng, "STRING", start, self.ind))
			elif self.cur in self.DIGITS:
				num = self.cur
				start = self.ind
				self.advance()
				while self.cur and self.cur in self.DIGITS:
					num += self.cur
					self.advance()
				toks.append(Token(int(num), "NUMBER", start, self.ind))
				self.back()
			elif self.cur in self.LETTERS:
				iden = ""
				start = self.ind
				while self.cur and self.cur in self.LETTERS + self.DIGITS + self.SPECIAL_KEYWORD_LETTERS:
					iden += self.cur
					self.advance()
				
				if iden not in self.keywords and True in [char in self.SPECIAL_KEYWORD_LETTERS for char in iden]:
					err = Error(self.code)
					err.error("Lexer", "Identifiers cant contain special characters.", start, len(iden))

				if iden not in self.keywords:
					toks.append(Token(iden, "IDENTIFIER", start, self.ind))
				else:
					toks.append(Token(iden, "KEYWORD", start, self.ind))
				self.back()
			elif self.cur == "#":
				while self.ind < len(self.code) and self.cur != "\n":
					self.advance()
				self.back()
			elif self.cur.isspace():
				pass
			else:
				err = Error(self.code)
				err.error("Lexer", f"Unexpected character '{self.cur}'", self.ind, 1)
			
			self.advance()
		return toks