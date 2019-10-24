import pandas


file_object = open("combine_dictionary.txt","r")
data2 = file_object.read().split('\n')



new_dictionary = sorted(data2)

file_object = open("sort_combine_dictionary.txt","w+")
for l in new_dictionary:
	y = str(l).replace('"','').strip().lower()
	x = y

	if (str(x) != "nan"):
		file_object.write(x + '\n')