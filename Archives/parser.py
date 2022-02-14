from error import Error
class Parser:
  def __init__(self, tokens, code):
    self.pos = 0
    self.tokens = tokens
    self.current_token = self.tokens[self.pos]
    self.code = code
  # good luck
  def get_next_token(self):
 
    token = self.tokens[self.pos]
    #print(token.type, self.pos, token.value)
    print(len(self.tokens))
    self.pos += 1
    
    return token
  
  def eat(self, tok_type):
    if self.current_token.type == tok_type:
      self.current_token = self.get_next_token()    
    else:
      self.error(self.current_token, tok_type)


  def expr(self): # you skip first token fard
    
    self.current_token = self.get_next_token()

    left = self.current_token
    self.eat("NUMBER")
    print("LEFT", left.value)
    #ill fix it by myself dont worry

    op = self.current_token

    if op.type == "PLUS":
      self.eat("PLUS")
    else:
      self.eat("MINUS")
    print("OP", op.value)
    
    right = self.current_token
    self.eat("NUMBER")

    if op.type == "PLUS":
      result = left.value + right.value
    else:
      result = left.value - right.value
    return result

    print(right)

  def error(self, tok, expect):
    err = Error(self.code)
    err.error("Parsing", f"Expected {expect}, but got \"{tok.value}\" instead!", tok.start, len(str(tok.value)))
		# I just realised error = would overwrite the imported thing


  


 #.   >>>>>>>>>>>>   new file maybe? lol dont need a new file
 #ye but lexer. Lexer is lexer. Exception is exception
#shoud I import the lexer file?
#ill move code

# Raise an error like:
# from lexer import Error
# err = Error(code) # code needs to be the code thats passed in to lexer
# err.error("Parser", "Error message", tok.start, len(str(tok.value)))