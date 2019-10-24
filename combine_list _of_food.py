import pandas


data = pandas.read_csv("Dish.csv", usecols=[1])
#print(data)

dictionary = data["name"].tolist()

print(len(dictionary))

file_object = open("dictionary_clean.txt","r")
data2 = file_object.read().split('\n')
#dictionary = data2;
dictionary.extend(data2 )
print(len(dictionary))


new_dictionary = sorted(dictionary)

file_object = open("combine_dictionary.txt","w+")
for l in new_dictionary:
	y = str(l).replace('"','').strip().lower()
	x = y

	if (str(x) != "nan"):
		file_object.write(x + '\n')