def astPrint(ast, level=0):
  print(" " * level + type(ast).__name__)
  level += 2

  for k, v in ast.__dict__.items():
    if type(v) not in [str, int, tuple, list]:
      print(f"{' ' * level}{k} =")
      astPrint(v, level + 2)
    elif type(v) == list:
      print(f"{' ' * level}{k} = [")
      
      for el in v:
        astPrint(el, level + 2)
      print(' ' * level + "]")
    else:
      print(f"{' ' * level}{k} = {v}")