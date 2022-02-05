print("START")
code = ""
def cleanup():
  with open("main.py", "r") as codefile:
    code = codefile.read().replace("	", "\t")

  #don't worry it only replaces tab

  #nonono that dangerous we copy pasta from main
  with open("main.py", "w") as newFile:
    newFile.write(code)
cleanup()

code =  """
=test

# hehe fard dis is a fardy comment

FARDING # begin program

refard FardFunction& # no arguments bcuz fard
			fard "fardy string"
			fardyNumber = 0
			fardyNumber += 1
			fard "%fardyNumber"

STOP_FARDING # end program
"""

from lexer import Lexer

lexer = Lexer(code)


class TokenFollower:
  def __init__(self, tokens):
    self.token = tokens
  def error(error,msg, linenum, expl):
    print("Congrats you got an error... In FardLang")
    
    print(f"{error} at {linenum}\n {expl}")
    print("should probably reconsider your skill level")
  def execute(self):
    #help me plz lol
    pass
    cTokNum = 0
    programStart = False
    for token in self.tokens:
      
      if (token.type != "KEYWORD" and token.value != "FARDING") and programStart == False:
        self.error("SyntaxError", 0, "Must start program with keyword 'FARDING.")
      else:
        programStart = True

      if token.type == "KEYWORD":
        if token.value=="refard": pass
          
      
      




print("\n".join(map(lambda x: x.__repr__(), lexer.tokenizeFard())))

#fine
"""
# hehe fard dis is a fardy comment

FARDING # begin program

refard FardFunction& # no arguments bcuz fard
			fard "fardy string"
			fardyNumber = 0
			fardyNumber += 1
			fard "%fardyNumber"

STOP_FARDING # end program
"""


