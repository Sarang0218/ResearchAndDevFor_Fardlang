class PrecedenceCalc:
  def __init__(self, tokens):
    self.tokens = tokens

  def infixToPostfix(self):
    fullOP = []
    im = ""

    op = {
      "*":2,
      "/":2,
      "+":1,
      "-":1,
    }