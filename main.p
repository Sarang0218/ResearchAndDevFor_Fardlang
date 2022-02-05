
code = ""
def cleanup():
  with open("main.py", "r") as codefile:
    code = codefile.read().replace("	", "\t")

  #don't worry it only replaces tab

  #nonono that dangerous we copy pasta from main
  with open("main.p", "w") as newFile:
    newFile.write(code)
cleanup()

code =  """
# hehe fard dis is a fardy comment

FARDING # begin program

refard FardFunction& # no arguments bcuz fard
			fard "fardy string"
			fardyNumber = 0
			fardyNumber += 1
			fard "%fardyNumber"

STOP_FARDING # end program
"""
class Token:
	def __init__(self, token, typef="identifer", start=0, end=0):
		self.value = token
		self.start = start
		self.end = end
		self.type = typef
		
	def __repr__(self):
		return f"\n-----------\nTOKEN (\n\tType:{self.type},\n\tValue:{self.value},\n\tfrom {self.start} to {self.end}\n)\n-----------"

class Lexer:
	def __init__(self, code):
		self.code = code
		self.ind = 0
		self.cur = self.code[self.ind] if self.ind < len(self.code) else None
		self.DIGITS = "01234567890"
		
	def advance(self):
		self.ind += 1
		self.cur = self.code[self.ind] if self.ind < len(self.code) else None
		
	def back(self):
		self.ind -= 1
		self.cur = self.code[self.ind] if self.ind >= 0 else None
#we don't need that much

	# we do.
	def tokenizeFard(self):
		toks = []
		while self.cur:
			if self.cur == '"':
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
				while self.cur in self.DIGITS:
					num += self.cur
					self.advance()
				toks.append(Token(strng, "NUMBER", start, self.ind))
				self.back()
			self.advance()
		return toks


lexer = Lexer(code)

print("\n".join(map(lambda x: x.__repr__(), lexer.tokenizeFard())))

#fine