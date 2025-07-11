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
    parser = argparse.ArgumentParser(description='分子筛选和聚类分析工具')
    parser.add_argument('--data', type=str, required=True, help='输入CSV文件路径，包含canonical_smiles列')
    parser.add_argument('--output', type=str, default='./datasets/finetuning/EP4/selected_molecules_from_clusters.csv', help='输出CSV文件路径 (默认: ./datasets/finetuning/EP4/selected_molecules_from_clusters.csv)')
    parser.add_argument('--n_clusters', type=int, default=100, help='聚类数量 (默认: 100)')
    parser.add_argument('--molecules_per_cluster', type=int, default=20, help='每个cluster选择的分子数量 (默认: 20)')

    return parser.parse_args()

def calculate(mol):
    """计算分子的各种物理化学性质"""
    mol_weight = Descriptors.MolWt(mol)  # 分子量
    logp = Descriptors.MolLogP(mol)  # 脂水分配系数
    h_donors = Descriptors.NumHDonors(mol)  # 氢键供体数
    h_acceptors = Descriptors.NumHAcceptors(mol)  # 氢键受体数
    qed = QED.qed(mol)  # 药物相似性评分
    tpsa = Descriptors.TPSA(mol)  # 拓扑极性表面积
    sa = compute_sa_score(mol)
    rot = Descriptors.NumRotatableBonds(mol)  # 可旋转键数
    return mol_weight, logp, h_donors, h_acceptors, qed, tpsa, sa, rot

def process_smiles_batch(smiles_batch):
    """处理一批SMILES字符串，用于多进程"""
    selected_smiles = []
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # 确保分子有效
            mol_weight, logp, h_donors, h_acceptors, qed, tpsa, sa, rot = calculate(mol)
            # 根据条件进行筛选
            if (450 <= mol_weight <= 600 and 5 <= logp <= 8 and 0 <= h_donors <= 5 and 0 <= h_acceptors <= 10 and qed >= 0.25 and tpsa <= 100 and sa <= 0.8 and rot < 12):
                selected_smiles.append(smiles)

    return selected_smiles

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.data):
        print(f"错误：输入文件 {args.data} 不存在！")
        return
    
    # 创建输出目录（如果不存在）
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取原始数据
    print(f"正在读取数据文件: {args.data}")
    data = pd.read_csv(args.data)

    # 根据物理化学性质筛选分子
    print("正在进行分子性质筛选...")
    
    # 获取CPU核心数，使用所有可用核心
    n_cores = cpu_count()
    print(f"使用 {n_cores} 个CPU核心进行并行处理")
    
    # 将SMILES数据分成批次
    smiles_list = data["canonical_smiles"].tolist()
    batch_size = len(smiles_list) // n_cores + 1
    smiles_batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
    
    selected_smiles = []
    
    # 使用多进程处理
    with Pool(processes=n_cores) as pool:
        # 使用tqdm显示进度
        results = list(tqdm(
            pool.imap(process_smiles_batch, smiles_batches),
            total=len(smiles_batches),
            desc="分子性质筛选",
            unit="批次"
        ))
        
        # 合并所有结果
        for batch_result in results:
            selected_smiles.extend(batch_result)

    selected_df = data[data["canonical_smiles"].isin(selected_smiles)]
    print(f"性质筛选后剩余分子数量: {len(selected_df)}")

    def Calc_AROM(mh):
        """计算分子中芳香环的数量"""
        m = Chem.RemoveHs(mh)  # 移除氢原子
        ring_info = m.GetRingInfo()  # 获取环信息
        atoms_in_rings = ring_info.AtomRings()  # 获取环中的原子
        num_aromatic_ring = 0
        for ring in atoms_in_rings:
            aromatic_atom_in_ring = 0
            for atom_id in ring:
                atom = m.GetAtomWithIdx(atom_id)
                if atom.GetIsAromatic():  # 检查原子是否为芳香性
                    aromatic_atom_in_ring += 1
            if aromatic_atom_in_ring == len(ring):  # 如果环中所有原子都是芳香性的
                num_aromatic_ring += 1
        return num_aromatic_ring

    df_threshold = pd.DataFrame()

    # 定义苯甲酸模式
    patt = Chem.MolFromSmiles("C1=CC=C(C=C1)C(=O)O")

    # 进一步筛选包含苯甲酸结构且满足条件的分子
    print("正在进行结构筛选...")
    for index, row in selected_df.iterrows():
        Atom = []
        gen_smile = row['canonical_smiles']

        mol = Chem.MolFromSmiles(gen_smile)
        hit_ats = mol.GetSubstructMatches(patt)  # 查找苯甲酸子结构

        for atom in mol.GetAtoms():
            Atom.append(atom.GetSymbol())

        num = Calc_AROM(mol)  # 计算芳香环数量

        # 筛选条件：包含苯甲酸结构、不含锂原子、芳香环数量小于4
        if len(hit_ats) > 0 and num < 4:
            df_threshold = df_threshold.append(selected_df.loc[index])

    # 重置索引
    df_threshold = df_threshold.reset_index(drop=True)
    print(f"结构筛选后剩余分子数量: {len(df_threshold)}")

    data = df_threshold
    smiles_column = 'canonical_smiles'
    smiles_data = data[smiles_column]

    def generate_ecfp(smiles_list):
        """生成ECFP分子指纹"""
        mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
        fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) for m in mols]  # 生成Morgan指纹
        return np.array(fps)

    # 生成分子指纹
    print("正在生成分子指纹...")
    fps = generate_ecfp(smiles_data)

    # 设置聚类参数
    n_clusters = args.n_clusters
    print(f"正在进行聚类分析，聚类数量: {n_clusters}")
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(fps)

    # 将聚类结果添加到数据框中
    data['cluster_id'] = clustering

    # 从每个cluster中随机选择指定数量的分子
    selected_molecules = []
    molecules_per_cluster = args.molecules_per_cluster

    print(f"正在从每个cluster中随机选择 {molecules_per_cluster} 个分子...")
    for cluster_id in range(n_clusters):
        # 获取当前cluster的所有分子
        cluster_data = data[data['cluster_id'] == cluster_id]
        
        if len(cluster_data) > 0:
            # 如果cluster中的分子数量大于等于指定数量，随机选择指定数量
            if len(cluster_data) >= molecules_per_cluster:
                selected_indices = random.sample(range(len(cluster_data)), molecules_per_cluster)
                selected_cluster_data = cluster_data.iloc[selected_indices]
            else:
                # 如果cluster中的分子数量少于指定数量，选择所有分子
                selected_cluster_data = cluster_data
                print(f"Cluster {cluster_id} 只有 {len(cluster_data)} 个分子，少于 {molecules_per_cluster} 个")
            
            selected_molecules.append(selected_cluster_data)

    # 合并所有选中的分子
    final_selected_df = pd.concat(selected_molecules, ignore_index=True)

    # 移除cluster_id列（如果需要的话）
    final_selected_df = final_selected_df.drop('cluster_id', axis=1)
    
    # 添加从1开始的index列
    final_selected_df.insert(0, 'index', range(1, len(final_selected_df) + 1))

    # 保存为新的CSV文件
    final_selected_df.to_csv(args.output, index=False)

    # 同时在data目录下创建同名文件夹并保存一份
    output_filename = os.path.basename(args.output)  # 获取输出文件名
    output_name_without_ext = os.path.splitext(output_filename)[0]  # 获取不带扩展名的文件名
    data_folder = os.path.join("data", output_name_without_ext)  # 在data目录下创建同名文件夹
    
    # 创建data目录下的同名文件夹
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(f"已创建文件夹: {data_folder}")
    
    # 在新文件夹中保存CSV文件
    additional_output_path = os.path.join(data_folder, output_filename)
    final_selected_df.to_csv(additional_output_path, index=False)

    print(f"已从 {n_clusters} 个cluster中选择了 {len(final_selected_df)} 个分子")
    print(f"结果已保存到: {args.output}")
    print(f"同时也保存到: {additional_output_path}")


if __name__ == "__main__":
    main()


