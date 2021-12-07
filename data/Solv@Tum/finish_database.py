from os import read
from pubchempy import get_compounds

dict = {}
cant_find = []
dict_smile = {}
path = 'F:\Project\DrugPP\data\Solv@Tum\\'
with open(path+'dict.txt', 'r') as reader:
    for line in reader:
        item = line.split(';')
        dict[item[0].replace('"', '')] = item[1].replace('\n', '')


def get_smile(name):
    if name in dict_smile:
        return dict_smile[name]
    compounds = get_compounds(name, 'name')
    compound = compounds[0] if compounds else None
    if compound is None:
        cant_find.append(name)
        return None
    else:
        dict_smile[name]=compound.isomeric_smiles
        return compound.isomeric_smiles


f = open(path+'Solv@Tum_reformated_unselect.txt', 'a')

with open(path+'not_down.txt', 'r') as reader:
    for line in reader:
        line = line.replace('"', '')
        line = line.replace('\n', '')
        solute, solvent, energy = line.split(';')
        if solute not in dict and solvent not in dict:
            continue
        solu_name = dict[solute] if solute in dict else solute
        solv_name = dict[solvent] if solvent in dict else solvent
        solu_smile = get_smile(solu_name)
        solv_smile = get_smile(solv_name)
        print(solu_name,solu_smile,solv_name,solv_smile)
        if solu_smile and solv_name:
            f.write(';'.join(
                [solu_name, solv_name, solu_smile, solv_smile,
                 str(energy)]) + '\n')

with open(path+'database.txt', 'r') as reader:
    for line in reader:
        f.write(line)

f.close()

if len(cant_find) != 0:
    print("Have Res Data!")
    for i in cant_find:
        print(i)
