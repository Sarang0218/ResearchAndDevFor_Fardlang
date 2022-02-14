import lexer
import data

text = """
fard "Hello World!"
"""
Language = lexer.Lexer("#")
Language.tokenize(text)
print(Language.tokens)
