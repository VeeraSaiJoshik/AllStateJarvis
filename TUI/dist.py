pencils_total = 5
students_total = 10
cap_max = 3

def comput_dist(arr: list[int]):
  count = 0
  print(arr)
  max_perm = cap_max
  if sum(arr) + max_perm > pencils_total : 
    max_perm = pencils_total - sum(arr)
  if len(arr) > students_total : return 0
  if len(arr) == students_total and sum(arr) == pencils_total: 
    return 1
  if sum(arr) > pencils_total : 
    return 0
  
  for i in range(max_perm + 1): 
    count += comput_dist(arr + [i])
  
  return count

print("we started")
print(comput_dist([]))