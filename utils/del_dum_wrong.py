import pandas as pd
import csv
import os
import argparse
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm  # 导入进度条库
from multiprocessing import Pool, cpu_count  # 导入多进程支持


def generate_scaffold(smiles):
    """Generate molecular scaffold"""
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    return scaffold


def process_single_molecule(row_data):
    """处理单个分子数据 - 用于多进程"""
    row, smiles_index = row_data
    smiles = row[smiles_index]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
    scaffold = generate_scaffold(canonical_smiles)
    # 返回处理后的行数据
    new_row = row.copy()
    new_row.insert(-1, canonical_smiles)
    new_row.append(scaffold)
    return new_row


def count_csv_len(fileName):
    """Count CSV file rows"""
    row_count = 0
    with open(fileName, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_count += 1
    return row_count


def process_molecules(input_file, output_file, temp_file='temporary.csv', n_processes=None):
    """Process molecular data: add canonical SMILES and scaffold using multiprocessing"""

    # 设置进程数，默认为CPU核心数
    if n_processes is None:
        n_processes = cpu_count()
    print(f"使用 {n_processes} 个进程进行并行处理")

    # 首先计算总行数用于进度条
    print("正在计算文件行数...")
    total_rows = count_csv_len(input_file) - 1  # 减去标题行
    print(f"总共需要处理 {total_rows} 行数据")

    # Step 1: Add canonical SMILES and scaffold using multiprocessing
    print("步骤1: 使用多进程添加规范SMILES和分子骨架...")
    
    # 读取所有数据
    with open(input_file, 'r', encoding="utf-8") as input_csv:
        reader = csv.reader(input_csv)
        header = next(reader)  # 读取标题行
        header.insert(-1, "canonical_smiles")
        header.append("generator_SMILES_scaffold")
        
        # 读取所有行数据
        all_rows = list(reader)
    
    # 准备多进程数据
    smiles_index = 1  # SMILES列索引
    process_data = [(row, smiles_index) for row in all_rows]
    
    # 使用多进程处理
    processed_rows = []
    with Pool(processes=n_processes) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_single_molecule, process_data),
            total=len(process_data),
            desc="并行处理分子数据",
            unit="行"
        ))
        
        # 收集有效结果
        for result in results:
            if result is not None:
                processed_rows.append(result)
    
    # 写入临时文件
    with open(temp_file, 'w', newline='', encoding="utf-8") as output_csv:
        writer = csv.writer(output_csv)
        writer.writerow(header)
        writer.writerows(processed_rows)

    # Step 2: Filter and remove duplicates
    print("步骤2: 过滤和去重...")
    df = pd.read_csv(temp_file)
    # Delete molecules containing *
    df = df[~df['canonical_smiles'].str.contains('\*')]
    # Remove duplicates based on canonical SMILES
    df = df.drop_duplicates(subset=['canonical_smiles'], keep='first')
    df.to_csv(output_file, index=False)

    print("4. Add canonical smiles and scaffold success!")

    # Clean up temporary file
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Count final rows
    row_count = count_csv_len(output_file)
    print(f"5. Total lines Used4screen is {row_count}")


def main():
    parser = argparse.ArgumentParser(description='Process molecular data: add canonical SMILES and scaffold, filter and deduplicate using multiprocessing')

    parser.add_argument('-i', '--input',
                        required=True,
                        help='Input CSV file path')

    parser.add_argument('-o', '--output',
                        required=True,
                        help='Output CSV file path')

    parser.add_argument('-t', '--temp',
                        default='temporary.csv',
                        help='Temporary file name (default: temporary.csv)')
    
    parser.add_argument('-p', '--processes',
                        type=int,
                        default=None,
                        help='Number of processes to use (default: CPU count)')

    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist!")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process molecular data
    process_molecules(args.input, args.output, args.temp, args.processes)
    print(f"Processing completed! Results saved to: {args.output}")


if __name__ == "__main__":
    main()
