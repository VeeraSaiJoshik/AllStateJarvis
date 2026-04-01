target = int(input())
def fib(n1, n2, n): 
  print(n, n1, n2, n1 + n2)
  if n == target : 
    return n1 + n2
  
  return fib(n2, n1 + n2, n + 1)

width = fib(0, 1, 2)

output = "#" * width + "\n"

print(output * width)
with open("square.txt", "w") as file: 
  file.write(output * width)