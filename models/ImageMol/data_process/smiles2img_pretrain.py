import argparse
import os
import shutil  # 添加shutil模块用于文件复制
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm import tqdm


def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png

        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================

    '''
    mol = Chem.MolFromSmiles(smis)  # 从SMILES字符串创建分子对象
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))  # 将分子转换为图像
    img.save(path)  # 保存图像到指定路径


def main():
    '''
    demo of raw_file_path:
        index,smiles
        1,CN(c1ccccc1)c1ccccc1C(=O)NCC1(O)CCOCC1
        2,CC[NH+](CC)C1CCC([NH2+]C2CC2)(C(=O)[O-])C1
        3,COCC(CNC(=O)c1ccc2c(c1)NC(=O)C2)OC
        ...
    :return:
    '''
    parser = argparse.ArgumentParser(description='Pretraining Data Generation for ImageMol')  # 创建参数解析器
    parser.add_argument('--dataroot', type=str, default="./datasets/pretraining/", help='data root')  # 数据根目录参数
    parser.add_argument('--dataset', type=str, default="data", help='dataset name, e.g. data')  # 数据集名称参数
    args = parser.parse_args()  # 解析命令行参数

    raw_file_path = os.path.join(args.dataroot, args.dataset, "{}.csv".format(args.dataset))  # 原始CSV文件路径
    img_save_root = os.path.join(args.dataroot, args.dataset, "processed/224")  # 图像保存根目录
    csv_save_path = os.path.join(args.dataroot, args.dataset, "{}_for_pretrain.csv".format(args.dataset))  # 处理后CSV保存路径
    error_save_path = os.path.join(args.dataroot, args.dataset, "error_smiles.csv")  # 错误SMILES保存路径

    # 创建processed目录（如果不存在）
    processed_dir = os.path.join(args.dataroot, args.dataset, "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # 将原始CSV文件复制到processed目录并重命名为BACE1_processed_ac.csv
    processed_csv_path = os.path.join(processed_dir, "{}_processed_ac.csv".format(args.dataset))
    if os.path.exists(raw_file_path) and not os.path.exists(processed_csv_path):
        shutil.copy2(raw_file_path, processed_csv_path)
        print(f"已将 {raw_file_path} 复制到 {processed_csv_path}")

    if not os.path.exists(img_save_root):  # 如果图像保存目录不存在则创建
        os.makedirs(img_save_root)

    df = pd.read_csv(raw_file_path)  # 读取原始CSV文件
    index, smiles = df["index"].values, df["smiles"].values  # 提取index和smiles列

    processed_ac_data = []  # 存储处理成功的数据
    error_smiles = []  # 存储处理失败的SMILES
    for i, s in tqdm(zip(index, smiles), total=len(index)):  # 遍历所有数据
        filename = "{}.png".format(i)  # 生成图像文件名
        img_save_path = os.path.join(img_save_root, filename)  # 图像保存路径
        try:
            loadSmilesAndSave(s, img_save_path)  # 将SMILES转换为图像并保存
            processed_ac_data.append([i, filename])  # 记录处理成功的数据
        except:
            error_smiles.append(s)  # 记录处理失败的SMILES

    processed_ac_data = np.array(processed_ac_data)  # 转换为numpy数组
    pd.DataFrame({
        "index": processed_ac_data[:, 0],  # index列
        "filename": processed_ac_data[:, 1],  # filename列
    }).to_csv(csv_save_path, index=False)  # 保存处理后的CSV文件

    if len(error_smiles) > 0:  # 如果有错误SMILES则保存
        pd.DataFrame({"smiles": error_smiles}).to_csv(error_save_path, index=False)


if __name__ == '__main__':
    main()
