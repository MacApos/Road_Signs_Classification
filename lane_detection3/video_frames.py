import re
import random

number = str(random.randint(2000, 20000))
print(number)

re.split(r'\d+', number)
