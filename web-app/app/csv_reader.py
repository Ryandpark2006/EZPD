text = open('text_copy.csv', 'r').read().splitlines()
cols = text[0].split(',')
data_groups = text[1].split(';')
print(len(cols), len(data_groups))