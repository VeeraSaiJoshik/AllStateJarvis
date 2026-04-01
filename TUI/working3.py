import math 

days31 = [1, 3, 5, 7, 9, 10, 12]
days30 = [4, 6, 9, 11]
days28 = [2]

class Date:
  def __init__(self, raw: str): 
    self.month = int(raw.split("/")[0])
    self.day = int(raw.split("/")[1])
    self.year = int(raw.split("/")[2])

    month_days = 0
    for m in range(self.month):
      if m in days31 : month_days += 31
      if m in days30 : month_days += 30
      if m in days28 : month_days += 28

    self.days_value = month_days + self.day

dates_raw = input()
dates = dates_raw.split(",")
date1 = Date(dates[0])
date2 = Date(dates[1])

print(math.fabs(date1.year - date2.year) * 365 + math.fabs(date1.days_value - date2.days_value))