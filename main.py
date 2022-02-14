
code = ""
def cleanup():
  with open("main.py", "r") as codefile:
    code = codefile.read().replace("	", "\t")

  #don't worry it only replaces tab

  #nonono that dangerous we copy pasta from main
  with open("main", "w") as newFile:
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

with open("main.gas", "r") as gas:
  code = gas.read()

code = code.replace("\t", "  ")

from lexer import Lexer
#from parser import Parser

lexer = Lexer(code)
tokens = lexer.tokenizeFard()


####################################################################
#                           PARSER TEST                            #
####################################################################
from fardparser import Parser
import astPrint
lexer = Lexer(code)
parser = Parser(lexer.tokenizeFard(), code)

ast = parser.parse()

# from compiler import Compiler
#
# compiler = Compiler(ast, code)

# compiler.compile()

# astPrint.astPrint(ast)
from interpreter import Interpreter, Scope
interpreter = Interpreter(ast, scope=Scope(parent=None, top=True))
interpreter.run()

#interpreter.currentScope.executeFunction("main")

#lexedMath = lexerMath.tokenizeFard()
#parser = Parser(lexedMath, math)
#print(parser.expr())
##################

#print("\n".join(map(lambda x: x.__repr__(), lexer.tokenizeFard())))

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
"""CODE FROM RANDOM SITE"""
"""

"""

# do you know howw to add precedence?
#well I've written a paper about it once

#you need to make a stack
#wait hold on imma get the pdf


#you can convert the code to reverse polish notation
#then you can calculate the code with a stack
#hold on