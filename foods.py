# traverse List of foods to get the different "Lists of " foods

html = requests.get('https://en.wikipedia.org/wiki/Lists_of_foods')
beautifulSoup = BeautifulSoup(html.text, 'lxml')
links = []
for i in beautifulSoup.find_all(name = 'li'):
  for link in i.find_all('a', href=True):
    title = link.get('title')
    href = link.get('href')
    data =[title,href]
    if title is not None and "List of " in title:
      links.append(data)


#now get the foods 

food_items = []

def get_food_items(main_title,link, writer):
  html = requests.get(link)
  beautifulSoup = BeautifulSoup(html.text, 'lxml')
  links = []
  for i in beautifulSoup.find_all(name = 'li'):
    for link in i.find_all('a', href=True):
      title = link.get('title')
      data =[main_title, title]
      if title is not None and "List of " not in title:
        food_items.append(data)
        writer.writerow(data)
        
with open ('document.csv', 'a') as fd:
  writer = csv.writer(fd)
  for link in links:
    get_food_items(link[0],'https://en.wikipedia.org' + link[1], writer)
  