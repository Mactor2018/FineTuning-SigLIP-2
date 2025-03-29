import json
import os
import warnings
warnings.filterwarnings("ignore")

import gc
import numpy as np
import pandas as pd
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import RandomOverSampler
import evaluate
from datasets import Dataset, Image, ClassLabel
from transformers import (
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomRotation,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    Resize,
    ToTensor
)

from PIL import Image as PILImage
from PIL import ImageFile
# 启用加载截断图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 添加logging和tqdm
import logging
from tqdm.auto import tqdm
import argparse
from transformers import AutoImageProcessor, SiglipForImageClassification

# 添加分布式训练支持
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建参数解析器
def parse_args():
    parser = argparse.ArgumentParser(description="SigLIP模型微调训练脚本")
    parser.add_argument("--data_path", type=str, required=True, help="JSON数据文件路径")
    parser.add_argument("--base_dir", type=str, default="/work/home/yinshb/yinshb/zjx/model_and_data/data", help="图像文件的基础目录")
    parser.add_argument("--output_dir", type=str, default="output/siglip2-birds", help="输出目录")
    parser.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch14-384", help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批量大小")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="评估批量大小")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test_size", type=float, default=0.3, help="测试集比例")
    parser.add_argument("--local_model_path", type=str, default=None, help="本地模型路径，如果提供则使用本地模型而不是远程下载")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载器的工作进程数")
    parser.add_argument("--sample_size", type=int, default=None, help="仅使用部分数据进行训练，如果设置则只使用指定数量的样本")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"], help="混合精度训练类型")
    parser.add_argument("--resize_size", type=int, default=None, help="调整图像大小的目标尺寸，默认使用模型的要求")
    parser.add_argument("--local_rank", type=int, default=-1, help="用于分布式训练的本地进程排名")
    parser.add_argument("--use_deepspeed", action="store_true", help="是否使用DeepSpeed进行训练")
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化分布式训练环境
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        args.local_rank = local_rank
    
    # 是否启用分布式训练
    is_distributed = args.local_rank != -1
    
    if is_distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logger.info(f"启动分布式训练: rank {rank} / {world_size}")
        # 只在主进程上打印信息
        if rank != 0:
            logger.setLevel(logging.WARNING)

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"加载数据: {args.data_path}")
    # 1. 加载JSON数据
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 如果指定了sample_size，则只使用部分数据
    if args.sample_size and args.sample_size < len(data):
        logger.info(f"使用样本量限制: {args.sample_size}，从 {len(data)} 个总样本中抽取")
        # 随机采样以保持类别分布
        np.random.shuffle(data)
        data = data[:args.sample_size]

    # 2. 提取图片路径和标签
    file_names = []
    labels = []

    for item in tqdm(data, desc="处理数据项", disable=not (not is_distributed or args.local_rank==0)):
        image_path = item["image"]
        # 获取最后一个对话作为标签
        label = item["conversations"][-1]["value"]
        
        file_names.append(image_path)
        labels.append(label)

    logger.info(f"总样本数: {len(file_names)}")

    # 3. 创建数据框
    df = pd.DataFrame({"image": file_names, "label": labels})
    logger.info(f"数据框大小: {df.shape}")

    # 查看前几条记录和标签分布
    logger.info("数据样例:")
    logger.info(df.head())
    logger.info("\n标签分布:")
    label_counts = df['label'].value_counts().head(10).to_dict()
    for label, count in label_counts.items():
        logger.info(f"{label}: {count}")

    # 4. 处理标签
    # 获取唯一标签列表
    unique_labels = df['label'].unique().tolist()
    logger.info(f"共有 {len(unique_labels)} 个不同类别")
    
    # 过滤掉只有一个样本的类别
    label_counts = df['label'].value_counts()
    # 找出至少有2个样本的标签
    valid_labels = label_counts[label_counts >= 2].index.tolist()
    # 统计被过滤掉的类别数
    filtered_labels = len(unique_labels) - len(valid_labels)
    logger.info(f"过滤掉 {filtered_labels} 个只有1个样本的类别")
    # 过滤数据集，只保留这些标签
    df = df[df['label'].isin(valid_labels)]
    logger.info(f"过滤后数据框大小: {df.shape}，保留了 {len(valid_labels)} 个类别")
    
    # 更新唯一标签列表
    unique_labels = valid_labels

    # 创建标签映射
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for idx, label in enumerate(unique_labels)}

    # 5. 将标签转换为ID
    df['label_id'] = df['label'].map(label2id)

    # 6. 创建数据集对象
    # 假设图片路径是相对于某个基础目录的
    base_dir = args.base_dir  # 从参数获取

    def get_image_path(relative_path):
        return os.path.join(base_dir, relative_path)

    # 构建数据集
    def create_dataset(dataframe):
        return Dataset.from_dict({
            "image_path": [get_image_path(path) for path in dataframe["image"]],
            "label": dataframe["label_id"].tolist()
        })

    # 7. 按照指定比例拆分数据集
    from sklearn.model_selection import train_test_split

    # 确保分层抽样
    logger.info(f"按照 {1-args.test_size}:{args.test_size} 比例拆分数据集...")
    train_df, test_df = train_test_split(
        df, 
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df['label']  # 确保标签分布一致
    )

    logger.info(f"训练集大小: {train_df.shape}")
    logger.info(f"测试集大小: {test_df.shape}")

    # 8. 创建数据集对象
    logger.info("创建数据集对象...")
    train_dataset = create_dataset(train_df)
    test_dataset = create_dataset(test_df)

    # 加载模型和处理器
    logger.info(f"加载模型: {args.model_name}")
    # 设置HF镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 根据是否提供本地模型路径决定加载方式
    # 修改处理器加载代码
    if args.local_model_path:
        logger.info(f"从本地路径加载模型: {args.local_model_path}")
        processor = AutoImageProcessor.from_pretrained(args.local_model_path, use_fast=True)
    else:
        processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)

    # 提取预处理参数
    image_mean, image_std = processor.image_mean, processor.image_std
    size = args.resize_size if args.resize_size else processor.size["height"]
    logger.info(f"图像大小设置为: {size}x{size}")

    # 修改训练转换为更高效的版本
    _train_transforms = Compose([
        Resize((size, size)),  # 调整大小
        ToTensor(),
        Normalize(mean=image_mean, std=image_std)
    ])

    # 定义验证转换
    _val_transforms = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std)
    ])

    # 9. 设置特征信息（改为流式加载）
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform, is_train=True):
            self.dataset = dataset
            self.transform = transform
            self.is_train = is_train
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image_path = item["image_path"]
            label = item["label"]
            
            try:
                # 优化的图像加载：先调整大小再转换，减少内存使用
                image = PILImage.open(image_path).convert("RGB")
                # 先调整大小，减少后续处理的内存需求
                if image.width > size or image.height > size:
                    image = image.resize((size, size), PILImage.BILINEAR)
                # 应用其他转换
                pixel_values = self.transform(image)
            except Exception as e:
                logger.warning(f"无法加载图像 {image_path}: {e}")
                # 创建一个黑色图像作为替代
                image = PILImage.new("RGB", (size, size), color=0)
                pixel_values = self.transform(image)
            
            return {
                "pixel_values": pixel_values,
                "labels": label
            }
    
    # 创建训练和测试数据集
    train_img_dataset = ImageDataset(train_dataset, _train_transforms, is_train=True)
    test_img_dataset = ImageDataset(test_dataset, _val_transforms, is_train=False)
    
    logger.info(f"训练集大小: {len(train_img_dataset)}")
    logger.info(f"测试集大小: {len(test_img_dataset)}")
    
    # 为分布式训练创建采样器
    train_sampler = DistributedSampler(train_img_dataset) if is_distributed else None
    test_sampler = DistributedSampler(test_img_dataset, shuffle=False) if is_distributed else None
    
    # 创建数据加载器，降低worker数量
    train_dataloader = DataLoader(
        train_img_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),  # 如果使用分布式采样器，则不需要在DataLoader中进行shuffle
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,  # 禁用pin_memory以减少内存使用
        persistent_workers=True if args.num_workers > 0 else False,  # 保持worker进程存活以加速迭代
        drop_last=True if is_distributed else False  # 在分布式训练中，确保每个批次具有相同的大小
    )
    
    test_dataloader = DataLoader(
        test_img_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    logger.info("数据加载器创建完成!")

    # 加载SigLIP模型用于分类
    if args.local_model_path:
        model = SiglipForImageClassification.from_pretrained(args.local_model_path, num_labels=len(unique_labels))
    else:
        model = SiglipForImageClassification.from_pretrained(args.model_name, num_labels=len(unique_labels))
    model.config.id2label = id2label
    model.config.label2id = label2id

    logger.info(f"可训练参数数量：{model.num_parameters(only_trainable=True) / 1e6:.2f}M")

    # 定义评估指标
    metric = evaluate.combine(["accuracy", "f1"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        
        return metric.compute(predictions=predictions, references=labels, average="weighted")

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=50,
        remove_unused_columns=False,
        save_strategy='epoch',
        load_best_model_at_end=True,
        save_total_limit=1,
        report_to="none",
        logging_strategy="steps",
        logging_steps=10,
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,
        fp16=args.mixed_precision == "fp16",  # 启用FP16训练
        bf16=args.mixed_precision == "bf16",  # 启用BF16训练
        gradient_accumulation_steps=1,  # 可以增加以减少内存使用
        gradient_checkpointing=True,  # 启用梯度检查点以减少内存使用
        optim="adamw_torch",
        # 分布式训练参数
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=500,
        deepspeed=os.path.join("configs", "deepspeed_config.json") if args.use_deepspeed else None
    )

    # 创建一个虚拟数据集，以便Trainer能够正确初始化
    dummy_train_dataset = Dataset.from_dict({
        "pixel_values": [np.zeros((3, size, size), dtype=np.float32)],
        "labels": [0]
    })
    
    dummy_eval_dataset = Dataset.from_dict({
        "pixel_values": [np.zeros((3, size, size), dtype=np.float32)],
        "labels": [0]
    })

    # 初始化训练器 - 使用数据加载器而不是数据集
    class CustomTrainer(Trainer):
        def get_train_dataloader(self):
            return train_dataloader
            
        def get_eval_dataloader(self, eval_dataset=None):
            return test_dataloader
            
        def get_test_dataloader(self, test_dataset=None):
            return test_dataloader
        
        def _set_sampler_epoch(self, epoch):
            # 在每个epoch开始时设置分布式采样器的epoch
            if is_distributed and hasattr(self.get_train_dataloader(), "sampler"):
                sampler = self.get_train_dataloader().sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)

    # 初始化训练器
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dummy_train_dataset,  # 提供虚拟数据集以满足Trainer要求
        eval_dataset=dummy_eval_dataset,    # 提供虚拟数据集以满足Trainer要求
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    
    # logger.info("[Skipped]训练前评估...")
    # initial_results = trainer.evaluate()
    # logger.info(f"训练前评估结果: {initial_results}")

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 训练后评估 - 只在主进程上执行
    if not is_distributed or args.local_rank == 0:
        logger.info("训练后评估...")
        final_results = trainer.evaluate()
        logger.info(f"训练后评估结果: {final_results}")

        # 保存最终模型
        model_path = os.path.join(args.output_dir, "final")
        trainer.save_model(model_path)
        processor.save_pretrained(model_path)
        logger.info(f"模型已保存到 {model_path}")

        # 保存标签映射
        with open(os.path.join(model_path, "label_mappings.json"), "w", encoding="utf-8") as f:
            json.dump({
                "id2label": id2label,
                "label2id": label2id
            }, f, ensure_ascii=False, indent=2)
        
        # 收集预测结果
        logger.info("生成预测...")
        all_preds = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="生成预测"):
                inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(model.device)
                
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 生成分类报告
        logger.info("分类报告:")
        class_names = [id2label[i] for i in range(len(unique_labels))]
        report = classification_report(all_labels, all_preds, target_names=class_names)
        logger.info("\n" + report)

        logger.info("训练和评估完成！")
    
    # 分布式训练环境清理
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 