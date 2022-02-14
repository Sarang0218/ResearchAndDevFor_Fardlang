
from error import Error

class Integer:
  def __init__(self, val):
    self.value = val
  
  def asString(self):
    return str(self.value)

class String:
  def __init__(self, val):
    self.value = val
  
  def asString(self):
    return self.value

class Null:
  def __init__(self):
    self.value = None
  
  def asString(self):
    return "null"

class Bool:
  def __init__(self, val):
    self.value = val
  
  def asString(self):
    if self.value: return "true"
    return "false"

class FunctionWrapper:
  def __init__(self, func, interp):
    self.func = func
    self.parScope = interp.currentScope
    self.funcScope = Scope(self.parScope)

  def execute(self):
    self.executor = Interpreter("nop")
    self.executor.currentScope = self.funcScope
    
    for node in self.func.body:
      self.executor.execute(node)

    self.funcScope = self.executor.currentScope # functions save their scope

class Scope:
  def __init__(self, parent=None, top=False):
    self.parent = parent
    self.top = top
    self.variables = {}
    self.functions = {}

  def executeFunction(self, name):
    if self.functions.get(name.value, None):
      self.functions[name.value].execute()
    else:
      row, col = Error.getRowCol(name)
      Error.generalError(f"No function named {name.value} at {row}:{col}.")
  
  def setFunction(self, name, body):
    self.functions[name] = body
  
  def set_variable(self, name, value):
    if self.top or self.variables.get(name, None):
      self.variables[name] = value
    else:
      self.parent.set_variable(name, value)
  
  def get_variable(self, name):
    if self.variables.get(name, None):
      return self.variables[name]
    else:
      if not self.top:
        return self.parent.get_variable(name)
      
      if self.top:
        Error.generalError(f"No variable named {name}")

from astPrint import astPrint
class Interpreter:
  def __init__(self, ast, scope=None):
    self.ast = ast
    
    if scope == None:
      scope = Scope()
    
    self.currentScope = scope

  def visit_BinaryExpression(self, expr):
    if "+" == expr.op:
      return self.execute(expr.left) + self.execute(expr.right)
    if "-" == expr.op:
      return self.execute(expr.left) - self.execute(expr.right)
    if "*" == expr.op:
      return self.execute(expr.left) * self.execute(expr.right)
    if "/" == expr.op:
      return self.execute(expr.left) / self.execute(expr.right)
  
  def visit_LogicalOp(self, logop):
    left = self.execute(logop.left)
    right = self.execute(logop.right)

    if logop.type.value == "==":
      return Bool(type(left) == type(right) and left.value == right.value)
    elif logop.type.value == ">=":
      return Bool(type(left) == type(right) and left.value >= right.value)
    elif logop.type.value == "<=":
      return Bool(type(left) == type(right) and left.value <= right.value)
    elif logop.type.value == "!=":
      return Bool(type(left) != type(right) and left.value != right.value)
    elif logop.type.value == "<":
      return Bool(type(left) == type(right) and left.value < right.value)
    elif logop.type.value == ">":
      return Bool(type(left) == type(right) and left.value > right.value)
  
  def visit_IfStatement(self, ifst):
    if self.execute(ifst.condition).value == False:
      return
    
    parScope = self.currentScope
    ifScope = Scope(parScope)
    executor = Interpreter("nop")
    executor.currentScope = ifScope
    
    for node in ifst.body:
      executor.execute(node)

  def visit_BoolLiteral(self, bo):
    if bo.value.value == "true": return Bool(True)
    return Bool(False)
  
  def visit_PrintKeyword(self, printk):
    print(self.execute(printk.value).asString())
  
  def visit_AccessVariable(self, var):
    return self.currentScope.get_variable(var.name.value)

  def visit_NumericLiteral(self, num):
    return Integer(num.value)
  
  def visit_VariableDefinition(self, var):
    self.currentScope.set_variable(
      var.name.value,
      self.execute(var.value)
    )
  
  def visit_CallFunction(self, func):
    self.currentScope.executeFunction(func)
    return Null()
  
  def visit_StringLiteral(self, string):
    return String(string.value)

  def visit_FunctionDefinition(self, func):
    name = func.name
    self.currentScope.setFunction(name, FunctionWrapper(func, self))

  def visit_Program(self, prog):
    for node in prog.body:
      self.execute(node)

  def execute(self, ast):
    func = getattr(self, f"visit_{type(ast).__name__}", None)
    if func:
      return func(ast)
    else:
      print(f"Can't visit node {type(ast).__name__} because it hasn't got a visit method")
      exit(1)
  
  def run(self):
    self.execute(self.ast)
