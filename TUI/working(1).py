dirty = input("Points: ")
lessDirty = ""
for l in dirty:
  if not l == '(' and not l == ')':
    lessDirty += l
cleaner = lessDirty.split(",")
points = []
for i in range((len(cleaner) - 1)//2):
  points.append([int(cleaner[i*2]), int(cleaner[i*2+1])])

max_points = 0
for i in range(len(points) - 1):
  for j in range(len(points)):
    if i == j:
      continue
    if points[j][0] == points[i][0]: continue
    roc = ((points[j][1] - points[i][1]) / (points[j][0] - points[i][0]))
    # roc * x + points[i][1]
    valid_pts = 0
    for point in points:
      if point == points[i] or point == points[j]:
        continue
      if points[i][1] - (roc * (points[i][0] - point[0])) == point[1] : 
        valid_pts += 1
    
    if valid_pts + 2 > max_points : 
      max_points = valid_pts + 2

print(max_points)