import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关模块
import numpy as np  # 导入数值计算模块
import pandas as pd  # 导入数据处理模块
import torch  # 导入PyTorch深度学习框架
import torch.nn as nn  # 导入神经网络模块
import torch.nn.parallel  # 导入并行计算模块
import torch.optim  # 导入优化器模块
import torch.utils.data  # 导入数据加载模块
import torchvision.transforms as transforms  # 导入图像变换模块
from dataloader.image_dataloader import ImageDataset, get_datasets  # 导入图像数据加载器
from model.cnn_model_utils import load_model  # 导入模型加载工具
from utils.public_utils import cal_torch_model_params, setup_device  # 导入公共工具函数


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PyTorch Implementation of ImageMol - Prediction Mode')

    # 基础设置
    parser.add_argument('--dataset', type=str, default="BBBP", help='数据集名称, 例如 BBBP, tox21, ...')
    parser.add_argument('--dataroot', type=str, default="./data_process/data/", help='数据根目录')
    parser.add_argument('--gpu', default='0', type=str, help='使用的GPU索引')
    parser.add_argument('--workers', default=2, type=int, help='数据加载工作进程数 (默认: 2)')

    # 评估设置
    parser.add_argument('--batch', default=128, type=int, help='小批次大小 (默认: 128)')
    parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='检查点路径 (默认: None)')
    parser.add_argument('--imageSize', type=int, default=224, help='输入图像的高度/宽度')
    parser.add_argument('--image_model', type=str, default="ResNet18", help='例如 ResNet18, ResNet34')
    parser.add_argument('--task_type', type=str, default="classification", choices=["classification", "regression"],
                        help='任务类型')
    parser.add_argument('--output_file', type=str, default="predictions.csv", help='输出预测结果的文件名')

    return parser.parse_args()


def load_filenames_for_prediction(image_folder, txt_file):
    """为预测模式加载文件名，不需要标签"""
    df = pd.read_csv(txt_file)  # 读取CSV文件
    
    # 检查是否有index列
    if "index" in df.columns:
        index = df["index"].values.astype(int)  # 提取索引值
    else:
        # 如果没有index列，使用行号作为索引
        index = df.index.values
    
    # 生成图像文件路径
    names = [os.path.join(image_folder, str(item) + ".png") for item in index]
    
    # 创建虚拟标签（用于数据加载器，实际不使用）
    dummy_labels = np.zeros((len(names), 1))  # 单任务分类的虚拟标签
    
    return names, dummy_labels, index


@torch.no_grad()
def predict_on_dataset(model, data_loader, device, task_type="classification"):
    """对数据集进行预测"""
    assert task_type in ["classification", "regression"]  # 确保任务类型有效

    model.eval()  # 设置模型为评估模式

    y_scores = []  # 存储预测得分
    sample_indices = []  # 存储样本索引
    
    for step, data in enumerate(data_loader):
        if len(data) == 3:  # 如果返回了索引
            images, _, indices = data
            sample_indices.extend(indices)
        else:
            images, _ = data
        
        images = images.to(device)  # 将图像数据移动到GPU

        with torch.no_grad():
            pred = model(images)  # 模型预测
            y_scores.append(pred)  # 添加预测结果

    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()  # 合并所有预测结果

    if task_type == "classification":
        # 对于分类任务，应用sigmoid函数得到概率
        y_prob = torch.sigmoid(torch.Tensor(y_scores)).numpy()
        return y_scores, y_prob, sample_indices
    elif task_type == "regression":
        return y_scores, y_scores, sample_indices  # 回归任务直接返回预测值


def main(args):
    """主函数"""
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 设置可见GPU

    # 获取数据路径
    args.image_folder, args.txt_file = get_datasets(args.dataset, args.dataroot, data_type="processed")
    args.verbose = True

    device, device_ids = setup_device(1)  # 设置设备

    # 架构名称
    if args.verbose:
        print('架构: {}'.format(args.image_model))

    ##################################### 加载数据 #####################################
    # 图像变换
    img_transformer_test = [transforms.CenterCrop(args.imageSize), transforms.ToTensor()]
    
    # 加载文件名（无需标签）
    names, dummy_labels, indices = load_filenames_for_prediction(args.image_folder, args.txt_file)
    names, dummy_labels = np.array(names), np.array(dummy_labels)
    
    print("加载了 {} 个样本进行预测".format(len(names)))
    
    # 根据加载的模型确定任务数量
    num_tasks = 1  # 默认单任务，后续会根据模型调整

    # 图像标准化
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # 创建数据集，返回索引用于追踪
    test_dataset = ImageDataset(names, dummy_labels, index=indices, 
                               img_transformer=transforms.Compose(img_transformer_test),
                               normalize=normalize, ret_index=True, args=args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch,
                                                  shuffle=False,
                                                  num_workers=args.workers,
                                                  pin_memory=True)

    ##################################### 加载模型 #####################################
    # 首先加载检查点以确定任务数量
    if args.resume and os.path.isfile(args.resume):
        print("=> 正在加载检查点 '{}'".format(args.resume))
        try:
            checkpoint = torch.load(args.resume)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            
            # 从最后一层确定任务数量
            for key in state_dict.keys():
                if 'fc.weight' in key:
                    num_tasks = state_dict[key].shape[0]
                    break
            
            print("检测到 {} 个任务".format(num_tasks))
        except Exception as e:
            print("加载检查点时出错: {}".format(e))
            return
    else:
        print("=> 在 '{}' 处未找到检查点".format(args.resume))
        return

    # 加载模型
    model = load_model(args.image_model, imageSize=args.imageSize, num_classes=num_tasks)

    # 加载权重
    try:
        model.load_state_dict(state_dict)
        print("=> 加载完成")
    except Exception as e:
        print("加载模型权重时出错: {}".format(e))
        return

    print("参数数量: {}".format(cal_torch_model_params(model)))
    model = model.cuda()  # 将模型移动到GPU
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # 多GPU并行

    ##################################### 预测 #####################################
    print("开始预测...")
    y_scores, y_prob, sample_indices = predict_on_dataset(model=model, 
                                                          data_loader=test_dataloader,
                                                          device=device, 
                                                          task_type=args.task_type)

    ##################################### 保存结果 #####################################
    # 创建结果DataFrame
    results = []
    for i, (idx, score, prob) in enumerate(zip(sample_indices if sample_indices else indices, y_scores, y_prob)):
        if num_tasks == 1:
            # 单任务
            if args.task_type == "classification":
                results.append({
                    'index': idx,
                    'predicted_score': score[0],
                    'predicted_probability': prob[0],
                    'predicted_label': 1 if prob[0] > 0.5 else 0
                })
            else:
                results.append({
                    'index': idx,
                    'predicted_value': score[0]
                })
        else:
            # 多任务
            result_dict = {'index': idx}
            for task_idx in range(num_tasks):
                if args.task_type == "classification":
                    result_dict[f'task_{task_idx}_score'] = score[task_idx]
                    result_dict[f'task_{task_idx}_probability'] = prob[task_idx]
                    result_dict[f'task_{task_idx}_label'] = 1 if prob[task_idx] > 0.5 else 0
                else:
                    result_dict[f'task_{task_idx}_value'] = score[task_idx]
            results.append(result_dict)

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 按预测得分/概率排序
    if args.task_type == "classification":
        if num_tasks == 1:
            results_df = results_df.sort_values('predicted_probability', ascending=False)
            print("结果按预测概率降序排列")
        else:
            # 多任务情况下，按第一个任务的概率排序
            results_df = results_df.sort_values('task_0_probability', ascending=False)
            print("结果按任务0的预测概率降序排列")
    else:
        if num_tasks == 1:
            results_df = results_df.sort_values('predicted_value', ascending=False)
            print("结果按预测值降序排列")
        else:
            results_df = results_df.sort_values('task_0_value', ascending=False)
            print("结果按任务0的预测值降序排列")

    # 保存结果
    output_path = args.output_file
    results_df.to_csv(output_path, index=False)
    print("预测结果已保存到: {}".format(output_path))
    
    # 显示前几个结果
    print("\n前10个预测结果:")
    print(results_df.head(10))

    print("\n预测完成! 总共预测了 {} 个样本".format(len(results_df)))


if __name__ == "__main__":
    args = parse_args()  # 解析命令行参数
    main(args)  # 运行主函数 