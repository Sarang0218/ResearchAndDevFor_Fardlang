import os

class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	RED = '\033[31m'
	ITALIC = '\033[3m'

class Error:
	def __init__(self, code):
		self.quotes = [
			"We should call you fard instead.",
			"should reconsider your programming skills", 
			"Congrats, you failed fardlang",
      "Your skills are fard",
      "\"It's not a bug, it's a feature!\"",
      "Your happiness = 1/My happiness",
      "Fard",
      "Awesome work guys! A new error! Oh wait, you have no friends",
		]
		self.code = code
	
	def warn(self, typef, msg, start, till_i):

		from random import choice

		lineno = self.code[:start].count("\n")
		col = start

		for l in self.code.split("\n")[:lineno]:
			col -= len(l) + 1

		line = self.code.split("\n")[lineno]
		beginLen = len(str(lineno)) + 3
		space = " " * (beginLen + col)
		under = "^" * till_i
		
		print(bcolors.WARNING, end="", flush=True)
		print(f"{bcolors.UNDERLINE}{typef} error!{bcolors.ENDC + bcolors.WARNING}\n  {msg}")
		print(f"	At {lineno}:{col}")
		print(f"\n{lineno} | {line}")
		print(f"{space}{under}")
		print("\n" + bcolors.ITALIC + choice(self.quotes))
		print(bcolors.ENDC, end="", flush=True)

	@classmethod
	def generalError(self, msg):
		print(bcolors.RED, end="", flush=True)
		print(msg)
		print(bcolors.ENDC, end="", flush=True)
		exit(1)
  
	@classmethod
	def getRowCol(self, tok):
		lineno = self.code[:tok.start].count("\n")
		col = tok.start

		for l in self.code.split("\n")[:lineno]:
			col -= len(l) + 1
		
		return lineno, col
	
	def error(self, typef, msg, start, till_i):

		from random import choice

		lineno = self.code[:start].count("\n")
		col = start

		for l in self.code.split("\n")[:lineno]:
			col -= len(l) + 1

		line = self.code.split("\n")[lineno]
		beginLen = len(str(lineno)) + 3
		space = " " * (beginLen + col)
		under = "^" * till_i
		
		print(bcolors.RED, end="", flush=True)
		print(f"{bcolors.UNDERLINE}{typef} error!{bcolors.ENDC + bcolors.RED}\n  {msg}")
		print(f"	At {lineno}:{col}")
		print(f"\n{lineno} | {line}")
		print(f"{space}{under}")
		print("\n" + bcolors.ITALIC + choice(self.quotes))
		print(bcolors.ENDC, end="", flush=True)
		exit(1)