adharm_list = ["0001","0010","0100",1000,9000,"0900","0090"]
target_state = str(3333)

adharm_list = [str(a) for a in adharm_list]
explored = []

def get_poss_nums(num: str):
  n = int(num)
  return (n + 1)%10, (n - 1)%10

def get_all_possible(this_thing: str):
  l = []
  for i in range(len(this_thing)): 
    n = this_thing[i]
    n1, n2 = get_poss_nums(n)
    temp = list(this_thing)
    temp2 = list(this_thing)

    temp[i] = str(n1)
    temp2[i] = str(n2)
    
    l.append("".join(temp))
    l.append("".join(temp2))
    
  final = []
  for l_asdf in l : 
    if l_asdf not in adharm_list: 
      if l_asdf == target_state: 
        print("we are done gng")
        return True
      adharm_list.append(l_asdf)
      final.append(l_asdf)
  return final

def dfs(nums: list[str], n = 0):
  f_all = []
  for num in nums : 
    frontier = get_all_possible(num)
    if frontier == True : 
      print(n + 1, n)
      return n
    f_all += frontier
  
  return dfs(f_all, n + 1)
  
print(dfs(["0000"]))