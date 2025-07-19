from rdkit.Chem import Descriptors, QED
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import AgglomerativeClustering
import random
import os
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from utils.sascorer import compute_sa_score



def parse_args():
    parser = argparse.ArgumentParser(description='Molecular screening and clustering analysis tool')
    parser.add_argument('--data', type=str, required=True, help='Input CSV file path containing canonical_smiles column')
    parser.add_argument('--output', type=str, default='./datasets/finetuning/EP4/selected_molecules_from_clusters.csv', help='Output CSV file path (default: ./datasets/finetuning/EP4/selected_molecules_from_clusters.csv)')
    parser.add_argument('--n_clusters', type=int, default=100, help='Number of clusters (default: 100)')
    parser.add_argument('--molecules_per_cluster', type=int, default=20, help='Number of molecules to select per cluster (default: 20)')

    return parser.parse_args()

def calculate(mol):
    """Calculate various physicochemical properties of molecules"""
    mol_weight = Descriptors.MolWt(mol)  # Molecular weight
    logp = Descriptors.MolLogP(mol)  # Lipophilicity (logP)
    h_donors = Descriptors.NumHDonors(mol)  # Number of hydrogen bond donors
    h_acceptors = Descriptors.NumHAcceptors(mol)  # Number of hydrogen bond acceptors
    qed = QED.qed(mol)  # Drug-likeness score
    tpsa = Descriptors.TPSA(mol)  # Topological polar surface area
    sa = compute_sa_score(mol)
    rot = Descriptors.NumRotatableBonds(mol)  # Number of rotatable bonds
    return mol_weight, logp, h_donors, h_acceptors, qed, tpsa, sa, rot

def process_smiles_batch(smiles_batch):
    """Process a batch of SMILES strings for multiprocessing"""
    selected_smiles = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Ensure molecule is valid
            mol_weight, logp, h_donors, h_acceptors, qed, tpsa, sa, rot = calculate(mol)
            # Filter based on conditions
            if (450 <= mol_weight <= 600 and 5 <= logp <= 8 and 0 <= h_donors <= 5 and 0 <= h_acceptors <= 10 and qed >= 0.25 and tpsa <= 100 and sa <= 0.8 and rot < 12):
                selected_smiles.append(smiles)

    return selected_smiles

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()

    # Check if input file exists
    if not os.path.exists(args.data):
        print(f"Error: Input file {args.data} does not exist!")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read original data
    print(f"Reading data file: {args.data}")
    data = pd.read_csv(args.data)

    # Filter molecules based on physicochemical properties
    print("Performing molecular property screening...")

    # Get CPU core count, use all available cores
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")

    # Split SMILES data into batches
    smiles_list = data["canonical_smiles"].tolist()
    batch_size = len(smiles_list) // n_cores + 1
    smiles_batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    
    selected_smiles = []

    # Use multiprocessing
    with Pool(processes=n_cores) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_smiles_batch, smiles_batches),
            total=len(smiles_batches),
            desc="Molecular property screening",
            unit="batch"
        ))

        # Merge all results
        for batch_result in results:
            selected_smiles.extend(batch_result)

    selected_df = data[data["canonical_smiles"].isin(selected_smiles)]
    print(f"Number of molecules remaining after property screening: {len(selected_df)}")

    def Calc_AROM(mh):
        """Calculate the number of aromatic rings in a molecule"""
        m = Chem.RemoveHs(mh)  # Remove hydrogen atoms
        ring_info = m.GetRingInfo()  # Get ring information
        atoms_in_rings = ring_info.AtomRings()  # Get atoms in rings
        num_aromatic_ring = 0
        for ring in atoms_in_rings:
            aromatic_atom_in_ring = 0
            for atom_id in ring:
                atom = m.GetAtomWithIdx(atom_id)
                if atom.GetIsAromatic():  # Check if atom is aromatic
                    aromatic_atom_in_ring += 1
            if aromatic_atom_in_ring == len(ring):  # If all atoms in ring are aromatic
                num_aromatic_ring += 1
        return num_aromatic_ring

    df_threshold = pd.DataFrame()

    # Define benzoic acid pattern
    patt = Chem.MolFromSmiles("C1=CC=C(C=C1)C(=O)O")

    # Further screening for molecules containing benzoic acid structure and meeting conditions
    print("Performing structural screening...")
    for index, row in selected_df.iterrows():
        Atom = []
        gen_smile = row['canonical_smiles']

        mol = Chem.MolFromSmiles(gen_smile)
        hit_ats = mol.GetSubstructMatches(patt)  # Find benzoic acid substructure

        for atom in mol.GetAtoms():
            Atom.append(atom.GetSymbol())

        num = Calc_AROM(mol)  # Calculate number of aromatic rings

        # Screening conditions: contains benzoic acid structure, no lithium atoms, aromatic rings < 4
        if len(hit_ats) > 0 and num < 4:
            df_threshold = df_threshold.append(selected_df.loc[index])

    # Reset index
    df_threshold = df_threshold.reset_index(drop=True)
    print(f"Number of molecules remaining after structural screening: {len(df_threshold)}")

    data = df_threshold
    smiles_column = 'canonical_smiles'
    smiles_data = data[smiles_column]

    def generate_ecfp(smiles_list):
        """Generate ECFP molecular fingerprints"""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]  # Generate Morgan fingerprints
        return np.array(fps)

    # Generate molecular fingerprints
    print("Generating molecular fingerprints...")
    fps = generate_ecfp(smiles_data)

    # Set clustering parameters
    n_clusters = args.n_clusters
    print(f"Performing clustering analysis with {n_clusters} clusters")
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(fps)

    # Add clustering results to dataframe
    data['cluster_id'] = clustering

    # Randomly select specified number of molecules from each cluster
    selected_molecules = []
    molecules_per_cluster = args.molecules_per_cluster

    print(f"Randomly selecting {molecules_per_cluster} molecules from each cluster...")
    for cluster_id in range(n_clusters):
        # Get all molecules in current cluster
        cluster_data = data[data['cluster_id'] == cluster_id]

        if len(cluster_data) > 0:
            # If cluster has more than or equal to specified number, randomly select specified number
            if len(cluster_data) >= molecules_per_cluster:
                selected_indices = random.sample(range(len(cluster_data)), molecules_per_cluster)
                selected_cluster_data = cluster_data.iloc[selected_indices]
            else:
                # If cluster has fewer molecules than specified, select all molecules
                selected_cluster_data = cluster_data
                print(f"Cluster {cluster_id} has only {len(cluster_data)} molecules, fewer than {molecules_per_cluster}")

            selected_molecules.append(selected_cluster_data)

    # Merge all selected molecules
    final_selected_df = pd.concat(selected_molecules, ignore_index=True)

    # Remove cluster_id column if needed
    final_selected_df = final_selected_df.drop('cluster_id', axis=1)

    # Add index column starting from 1
    final_selected_df.insert(0, 'index', range(1, len(final_selected_df) + 1))

    # Save as new CSV file
    final_selected_df.to_csv(args.output, index=False)

    # Also create a folder with the same name in data directory and save a copy
    output_filename = os.path.basename(args.output)  # Get output filename
    output_name_without_ext = os.path.splitext(output_filename)[0]  # Get filename without extension
    data_folder = os.path.join("data", output_name_without_ext)  # Create folder with same name in data directory

    # Create folder with same name in data directory
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"Created folder: {data_folder}")

    # Save CSV file in new folder
    additional_output_path = os.path.join(data_folder, output_filename)
    final_selected_df.to_csv(additional_output_path, index=False)

    print(f"Selected {len(final_selected_df)} molecules from {n_clusters} clusters")
    print(f"Results saved to: {args.output}")
    print(f"Also saved to: {additional_output_path}")


if __name__ == "__main__":
    main()


