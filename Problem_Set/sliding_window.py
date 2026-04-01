import math

numbers = [12234, 98765, 43210, 12345, 67890, 54321, 11111, 22222, 33333, 44444]
window = 4

sums = []

for i in range(len(numbers) - window + 1):
  sums.append(round(sum(numbers[i: i + window])/window, 2))

print([float(str(sum) + "0") for sum in sums])