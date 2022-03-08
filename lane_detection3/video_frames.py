import re
import glob
import random

number = random.randint(2000, 20000)

fold = 1

example = r'my_vid\%d\*.jpg' % number

path = r'C:\Nowy folder\10\Praca\Datasets\tu-simple\TEST\*.jpg'
images = glob.glob(path)



print(images)

print(re.split(r'\d+', example))


