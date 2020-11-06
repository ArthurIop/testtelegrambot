from random import choice
from string import ascii_letters
index = 0 
for i in range (100):
	index +=1
	f = open('C:\\Users\\egiazaryan\\Desktop\\text-library\\000000'+str(index)+'.txt','w')
	text = ''.join(choice(ascii_letters) for i in range(2500000))
	f.write(text)
	f.close()