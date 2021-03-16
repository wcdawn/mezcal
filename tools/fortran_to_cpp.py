import re # regex

fortran_format = re.compile('(\d+)\*(.*)')

def expand(count, string):
    return int(count)*(str(string) + ' ')

def convert(string):
    match = fortran_format.search(string)
    return expand(int(match.group(2)), match.group(3))

fname = 'vvercan.inp'
#fname = 'vver238cpp.xs'

f = open(fname, 'r')
data = f.readlines()
f.close()

for i in range(len(data)):
    match = fortran_format.search(data[i])
    if (match is not None):
        replacement = expand(match.group(1), match.group(2))
        data[i] = data[i].replace(match.group(), replacement)

f = open(fname, 'w')
f.writelines(data)
f.close()
