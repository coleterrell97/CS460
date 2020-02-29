file = open("u-item.item", "r")

lines = file.readlines()
for i in lines:
	i = i.rstrip("\n")
	i = i.split("|")
	newArray = i[5:]
	#print(newArray)