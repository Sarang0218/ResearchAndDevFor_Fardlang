from error import Error, bcolors
from fardparser import StringLiteral, BinaryExpression, NumericLiteral, PrintKeyword, AccessVariable, VariableDefinition, FunctionDefinition
from lexer import Token

# NumericLiteral StringLiteral BinaryExpression FunctionDefinition PrintKeyword
# VariableDefinition AccessVariable

class Compiler:
  def __init__(self, ast, code):
    self.ast = ast
    self.alreadyDefined = []
    self.code = code
    self.variableTypes = {}
  
  def guessType(self, node):
    if type(node) == BinaryExpression:
      typel = self.guessType(node.left)
      typer = self.guessType(node.right)
      if typel != typer:
        error = Error(self.code)
        
        start = 0
        end = 0

        if type(node.left) == AccessVariable: start = node.left.name.start
        else: start = node.left.start

        if type(node.right) == AccessVariable: end = node.right.name.end
        else: end = node.right.end
        error.error("Compiler", f"Can't add 2 different types.\n  ({typel} {node.op} {typer})", start, end - start)
      else:
        return typel
    
    if type(node) == NumericLiteral: return "int"
    if type(node) == StringLiteral: return "string"
    if type(node) == AccessVariable:
      if not self.variableTypes.get(node.name.value, None):
        error = Error(self.code)
        error.error("Compiler", f"Undefined variable {node.name.value}.", node.name.start, len(node.name.value))
      return self.variableTypes[node.name.value]
    
    error = Error(self.code)
    error.generalError(f"Compiler error! Couldn't guess type.")

  def compileNode(self, node):
    if type(node) == NumericLiteral:
      return str(node.value)
    elif type(node) == StringLiteral:
      return '"' + node.value + '"'
    elif type(node) == BinaryExpression:
      return f"{self.compileNode(node.left)} {node.op} {self.compileNode(node.right)}"
    elif type(node) == FunctionDefinition:
      return "void %s() {%s}" % (node.name, self.compileBody(node.body))
    elif type(node) == PrintKeyword:
      return f"cout << {self.compileNode(node.value)} << endl;"
    elif type(node) == AccessVariable:
      return node.name.value
    elif type(node) == VariableDefinition:
      if node.name.value not in self.alreadyDefined:
        self.alreadyDefined.append(node.name.value)
        self.variableTypes[node.name.value] = self.guessType(node.value)

        typef = self.guessType(node.value)
        return f"{typef} {node.name.value} = {self.compileNode(node.value)};"
      else:
        typef = self.guessType(node.value)

        if self.variableTypes[node.name.value] != typef:
          error = Error(self.code)
          error.error("Compiler", "Can't switch variable type after declaration", node.name.start, node.name.end - node.name.start)
        
        return f"{node.name.value} = {self.compileNode(node.value)};"
    elif type(node) == Token:
      if node.type == "IDENTIFIER":
        return node.value
    else:
      error = Error(self.code)
      error.generalError("Compiler error!")

  def compileBody(self, l):
    s = ""
    for i in l: s += self.compileNode(i)
    return s

  def compile(self):

    code = """
#include <iostream>
using namespace std;

int main() {
"""
    for node in self.ast.body:
      code += "  " + self.compileNode(node) + "\n"
    
    code += "  return 0;\n}"

    filename = "main"
    with open("output/" + filename + ".cpp", "w") as f:
      f.write(code)
    
    import os
    import time
    import subprocess

    folder = "/".join(__file__.split("/")[:-1]) + "/output/"
    print(bcolors.WARNING + "Compiling your program." + bcolors.ENDC)
    os.system(
      f"g++ -o {folder + 'main'} {folder}{filename}.cpp"
    )
    print(bcolors.OKGREEN + "Successfully compiled your program." + bcolors.ENDC)

    os.remove(folder + filename + ".cpp")

    start = time.time()
    print(bcolors.WARNING + "Running your program..." + bcolors.ENDC)
    os.system(folder + filename)
    print(bcolors.OKGREEN + f"Successfully ran your program in {bcolors.OKCYAN}{round(time.time() - start, 3)}s" + bcolors.ENDC)