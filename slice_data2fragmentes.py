#!/usr/bin/env python
#  coding=utf-8

import argparse
import json

import pyspark.sql as ps
import pyspark.sql.functions as psf

import models.actions as ma
import utils.log as ul
import utils.chem as uc
import utils.spark as us
import utils.scaffold as usc
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import os


class SliceDB(ma.Action):

    def __init__(self, input_path, output_path, enumerator, max_cuts, partitions, logger=None):
        ma.Action.__init__(self, logger)

        self.input_path = input_path
        self.output_path = output_path
        self.enumerator = enumerator
        self.max_cuts = max_cuts
        self.partitions = partitions

    def run(self):
        def _enumerate(row, max_cuts=self.max_cuts, enumerator=self.enumerator):
            fields = row.split("\t")
            smiles = fields[0]
            mol = uc.to_mol(smiles)
            out_rows = []
            if mol:
                for cuts in range(1, max_cuts + 1):
                    try:
                        for sliced_mol in enumerator.enumerate(mol, cuts=cuts):

                            scaff_smi, dec_smis = sliced_mol.to_smiles()
                            dec_smis = [smi for num, smi in sorted(dec_smis.items())]

                            out_rows.append(ps.Row(
                                scaffold=scaff_smi,
                                decorations=dec_smis,
                                smiles=uc.to_smiles(mol),
                                cuts=cuts
                            ))
                    except Exception as e:
                        pass
            return out_rows
        enumeration_df = SPARK.createDataFrame(
            SC.textFile(self.input_path)
            .repartition(self.partitions)
            .flatMap(_enumerate))\
            .groupBy("scaffold", "decorations")\
            .agg(psf.first("cuts").alias("cuts"), psf.first("smiles").alias("smiles"))\
            .persist()

        self._log("info", "Obtained %d sliced molecules", enumeration_df.count())

        if self.output_path:
            enumeration_df.write.parquet(self.output_path)
        return enumeration_df


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Slices the molecules a given way.")
    parser.add_argument("--input_smiles_path", "-i", help="Path to the input file with molecules in SMILES notation.", type=str, required=True)
    parser.add_argument("--output-parquet-folder", "-o", help="Path to the output Apache Parquet folder.", type=str)
    parser.add_argument("--output_smiles_path", "-u", help="Path to the output SMILES file.", type=str)
    parser.add_argument("--max-cuts", "-c", help="Maximum number of cuts to attempts for each molecule [DEFAULT: 2]", type=int, default=2)
    parser.add_argument("--slice-type", "-s", help="Kind of slicing performed TYPES=(recap, hr) [DEFAULT: hr]", type=str, default="hr")
    parser.add_argument("--num-partitions", "--np", help="Number of Spark partitions to use (leave it if you don't know what it means) [DEFAULT: 1000]", type=int, default=1000)
    parser.add_argument("--conditions-file", "-f", help="JSON file with the filtering conditions for the scaffolds and the decorations.", type=str)
    parser.add_argument("--trainModel", help="The slice rule(not limit scaffold in train and limit scaffold in Inference) in Train or Inference. default is Train", action="store_true", default=False)

    return parser.parse_args()


def _to_smiles_rows(row):
    return "{}\t{}\t{}".format(row["scaffold"], ";".join(row["decorations"]), row["smiles"])

def canonical_smiles(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def get_scaffold(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    scaffold = canonical_smiles(scaffold)
    return scaffold


def main():
    """Main function."""
    args = parse_args()

    scaffold_conditions = None
    decoration_conditions = None
    # use the conditions.json file if provided
    if args.conditions_file:
        with open(args.conditions_file, "r") as json_file:
            data = json.load(json_file)
            if "scaffold" in data:
                scaffold_conditions = data["scaffold"]
            if "decoration" in data:
                decoration_conditions = data["decoration"]
    # define the slice rules
    enumerator = usc.SliceEnumerator(usc.SLICE_SMARTS[args.slice_type], scaffold_conditions, decoration_conditions)
    # begin slice
    slice_db_action = SliceDB(args.input_smiles_path, args.output_parquet_folder, enumerator, args.max_cuts, args.num_partitions, LOG)
    slice_df = slice_db_action.run()

    # 得到output_smiles_path'/'之前的路径
    os.system('mkdir -p {}'.format(os.path.dirname(args.output_smiles_path)))
    os.system('mkdir -p {}'.format(os.path.dirname(args.output_smiles_path) + '/models'))

    if args.output_smiles_path:
        with open(args.output_smiles_path, "w+") as smiles_file:
            for row in slice_df.rdd.map(_to_smiles_rows).toLocalIterator():
                smiles_file.write("{}\n".format(row))

LOG = ul.get_logger(name="slice_db")
SPARK, SC = us.SparkSessionSingleton.get("slice_db")
if __name__ == "__main__":
    main()
