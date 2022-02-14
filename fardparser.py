#what is this lol
#HELLO?
# o ok
#but we already have parser 
#okok
# good luck



# fard parser
# I want to generate javascript with fard
# like
# fard "hi" -> console.log('hi')

SampleAST = {
  "ok do that"
}

# yea like this
#hm

#this is crazy lol
# this is how asts look
# yea
# but should probably make it easier to Generate
# IE:  adding easy to customize parameters (make a new key called "params" and have a list of customizable parameters. )
# Include more information I guess
#raw characters, (original value from lexer)
#Nononono don't need indexes lmao
#ok I think this is fine for now.
#why not just have an instance of the token?
# too simple
# me no like
#bye bye parser :)
# rip parser
# Simple is better :D
#heh my parser is still breaking

# no it can easily be generated I only need precedence
# hm
# Ive got an ok idea...

#group the expressions into tokens.
# then, we can covert the tokens by precedence into postfix
#postfix has no precedence B)
#Time to implement converter
# B)))))))))))))))))))))))))))))))))))))))))))))))))


# are you here?
#if so, find a way to group expressions

#bye ill work on the precedence calc :)
Ast = {
  "body": [
    {
      "type": "BinaryExpression",
      "start": 1,
      "end": 3,
      "left": {
        "type": "NumericLiteral",
        "value": 3
      },
      "right": {
        "type": "BinaryExpression",
        "left": {
          "type": "NumericLiteral",
          "value": 3
        },
        "right": {
          "type": "NumericLiteral",
          "value": 3
        },
        "op": "*"
      },
      "op": "+"
    }
  ]
}

from error import Error

class NumericLiteral:
  def __init__(self, tok):
    self.value = tok.value
    self.start = tok.start
    self.end = tok.end

  def __repr__(self):
    return f"Number({self.value})"

class StringLiteral:
  def __init__(self, tok):
    self.value = tok.value
    self.start = tok.start
    self.end = tok.end

  def __repr__(self):
    return f"String({self.value})"

class BinaryExpression:
  def __init__(self, left, op, right):
    self.left = left
    self.op = op.value
    self.right = right
  
  def __repr__(self):
    return f"BinaryExpr({self.left}, {self.op}, {self.right})"

class FunctionDefinition:
  def __init__(self, name, body):
    self.name = name
    self.body = body

class IfStatement:
	def __init__(self, cond, body):
		self.condition = cond
		self.body = body

class PrintKeyword:
  def __init__(self, value):
    self.value = value

class VariableDefinition:
  def __init__(self, name, value):
    self.name = name
    self.value = value

class AccessVariable:
  def __init__(self, name):
    self.name = name

class Program:
  def __init__(self):
    self.body = []

class CallFunction:
  def __init__(self, n):
    self.name = n

class LogicalOp:
  def __init__(self, left, comp, right):
    self.left = left
    self.type = comp
    self.right = right

class BoolLiteral:
  def __init__(self, val):
    self.value = val

class Parser:
  def __init__(self, toks, code):
    self.toks = toks
    self.code = code
    self.ind = 0
    self.cur = self.toks[self.ind]
    self.err = Error(code)

  def advance(self):
    self.ind += 1
    self.cur = self.toks[self.ind] if self.ind < len(self.toks) else None

  def peek(self):
    return self.toks[self.ind + 1] if self.ind + 1 < len(self.toks) else None

  def logicalOp(self):
    if self.peek() and self.peek().type == "LOGICAL_OP":
      self.toks[self.ind + 1].type = "PROCESSING_LOGICAL_OP"
      left = self.primary()
      op = self.eat("PROCESSING_LOGICAL_OP")
      right = self.primary()

      return LogicalOp(left, op, right)
    
    self.err.error("Syntax", "Invalid syntax.", self.cur.start, len(self.value.value))

  def primary(self):

    if self.peek() and self.peek().type == "LOGICAL_OP":
      return self.logicalOp()

    elif self.cur.type == "BOOL":
      return BoolLiteral(self.eat("BOOL"))

    elif self.cur.type == "NUMBER":
      return NumericLiteral(self.eat("NUMBER"))

    elif self.cur.type == "STRING":
      return StringLiteral(self.eat("STRING"))

    elif self.cur.type == "IDENTIFIER":
      if self.peek() and self.peek().type == "LPAREN":
        n = self.eat("IDENTIFIER")
        self.eat("LPAREN")
        self.eat("RPAREN")
        return CallFunction(n)
      
      return AccessVariable(self.eat("IDENTIFIER"))

    else:
      self.eat("LPAREN")
      expr = self.additive()
      self.eat("RPAREN")
      return expr

  def multiplicative(self):
    left = self.primary()

    if self.cur and self.cur.type in ["TIMES", "DIV"]:
      op = None

      if self.cur.type == "TIMES":
        op = self.eat("TIMES")
      else:
        op = self.eat("DIV")
      
      right = self.multiplicative()

      return BinaryExpression(left, op, right)
    
    return left

  def additive(self):
    left = self.multiplicative()

    if self.cur and self.cur.type in ["PLUS", "MINUS"]:
      op = None
      if self.cur.type == "PLUS":
        op = self.eat("PLUS")
      else:
        op = self.eat("MINUS")
      
      right = self.additive()

      return BinaryExpression(left, op, right)
    
    return left

  def eat(self, typef):
    if self.cur and self.cur.type == typef:
      cur = self.cur
      self.advance()
      return cur
      
    else:
      self.err.error("Parser", f"Was expecting type {typef} but got {self.cur.type}.", self.cur.start, len(self.cur.value))

  def makeFunction(self):
    self.eat("KEYWORD")
    name = self.eat("IDENTIFIER")
    
    body = []

    while True:
      if self.cur.type == "KEYWORD" and self.cur.value == "farded":
        break
      body.append(self.expr())
    
    self.eat("KEYWORD")
    
    return FunctionDefinition(name.value, body)
	
  def makeIfStatement(self):
    self.eat("KEYWORD")
    cond = self.logicalOp()
    
    body = []

    while True:
      if self.cur.type == "KEYWORD" and self.cur.value == "farded":
        break
      body.append(self.expr())
    
    self.eat("KEYWORD")
    
    return IfStatement(cond, body)

  def expr(self):

    if self.cur.type == "KEYWORD" and self.cur.value == "refard":
      return self.makeFunction()
		
    elif self.cur.type == "KEYWORD" and self.cur.value == "fards?":
      return self.makeIfStatement()
    
    elif self.cur.type == "KEYWORD" and self.cur.value == "fard":
      self.eat("KEYWORD")
      value = self.additive()
      return PrintKeyword(value)
    
    elif self.cur.type == "IDENTIFIER":
      name = self.eat("IDENTIFIER")

      if self.cur.type == "EQUALS":
        self.eat("EQUALS")
        value = self.additive()

        return VariableDefinition(name, value)
    
      elif self.cur.type == "VARIABLE_MODIFY":
        op = self.eat("VARIABLE_MODIFY")

        op.value = op.value[:1]

        value = self.additive()
        return VariableDefinition(
          name,
          BinaryExpression(
            AccessVariable(name),
            op,
            value
          )
        )

    return self.additive()

  def parse(self):

    prog = Program()

    while self.cur:
      prog.body.append(self.expr())

    return prog
