import pandas as pd
import csv
import os
import csv
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold

def count_csv_len(fileName):
    row_count = 0
    with open(fileName, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_count += 1
    return row_count



input_file_name = 'data/results/EP4_inference_generated.csv'
output_file_name = 'data/results/EP4_inference_generated_filter.csv'

temporary_file_name = 'temporary.csv'



with open(input_file_name, 'r', encoding="utf-8") as input_csv:
    reader = csv.reader(input_csv)
    header = next(reader)
    header.insert(-1, "canonical_smiles")
    header.append("generator_SMILES_scaffold")

    with open(temporary_file_name, 'w', newline='', encoding="utf-8") as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow(header)

        for row in reader:
            smiles = row[1]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            scaffold = generate_scaffold(canonical_smiles)
            row.insert(-1, canonical_smiles)
            row.append(scaffold)
            writer.writerow(row)


df = pd.read_csv(temporary_file_name)
# delete *
df = df[~df['canonical_smiles'].str.contains('\*')]
df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')
df.to_csv(output_file_name, index=False)


print("4. Add canonical smiles and scaffold success!")
os.remove(temporary_file_name)

# 4、Count the len of csv
row_count = count_csv_len(output_file_name)
print("5. Total lines Used4screen is ", row_count)
