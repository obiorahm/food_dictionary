import pandas

data = pandas.read_csv("new_document.csv", usecols=[1])
print(data)

dictionary = data["astrebla pectinata"].tolist()

new_dictionary = sorted(dictionary)

file_object = open("dictionary.txt","w+")
for l in new_dictionary:
	x = str(l)
	if (str(x) != "nan"):
		file_object.write(x + '\n')

