from pubchempy import get_compounds
from tqdm import tqdm
import time

ok = set()
nok = set()
need = {'C', 'H', 'N', 'O', 'F', 'P', 'Cl', 'S', 'Br', 'I'}
data = []
res = []
dict = {
    'DICHLOROETHANE': '1,2-DICHLOROETHANE',
    'HEPTYLACETAT': 'Heptyl acetate'
}


def judge(name):
    name = dict[name] if name in dict else name
    if name in ok:
        return True
    if name in nok:
        return False
    compounds = get_compounds(name, 'name')
    compound = compounds[0] if compounds else None
    if compound:
        tmp = set(compound.elements)
        if not tmp.difference(need):
            ok.add(name)
            return True
        else:
            nok.add(name)
            return False
    else:
        print("Can't get element of " + name)
        res.append()
        return False


f = open('Solv@Tum_reformated.txt', 'w')
with open('Solv@Tum_reformated_unselect.txt', 'r') as reader:
    first_line = True
    cnt = 1
    bar = tqdm(reader)
    for line in bar:
        if first_line:
            f.write(line)
            first_line = False
            continue
        lines = line.split(';')
        solute = lines[0]
        solvent = lines[1]
        if judge(solute) and judge(solvent):
            f.write(line)
        bar.set_description("Processing {}".format(cnt))
        cnt += 1

f.close()
f = open('res.txt', 'w')
f.write('\n' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '\n')
print("{} can't find".format(len(res)))
for line in res:
    f.write(line)
f.close()