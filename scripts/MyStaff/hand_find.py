from pubchempy import get_compounds

ori = "mcresol"
chem = "m-cresol"
compounds = get_compounds(chem, 'name')
compound = compounds[0] if compounds else None
if compound:
    f = open('/users10/xzluo/DrugDP/DrugPP/data/MNSOL/full_dict.csv', 'a')
    add_text = ';'.join([ori, compound.isomeric_smiles, chem]) + '\n'
    f.write(add_text)
    print(add_text)
else:
    print("Can't Get formula of " + chem)
