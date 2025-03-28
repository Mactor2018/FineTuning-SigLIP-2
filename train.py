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
    parser.add_argument("--output_dir", type=str, default="checkpoints/siglip2-finetune", help="输出目录")
    parser.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch14-384", help="预训练模型名称")
    parser.add_argument("--batch_size", type=int, default=16, help="训练批量大小")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="评估批量大小")
    parser.add_argument("--learning_rate", type=float, default=2e-6, help="学习率")
    parser.add_argument("--num_epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="权重衰减")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test_size", type=float, default=0.3, help="测试集比例")
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logger.info(f"加载数据: {args.data_path}")
    # 1. 加载JSON数据
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 提取图片路径和标签
    file_names = []
    labels = []

    for item in tqdm(data, desc="处理数据项"):
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
            "image": [get_image_path(path) for path in dataframe["image"]],
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

    # 9. 添加图像加载功能
    def load_and_transform_images(examples, split="train"):
        images = []
        error_count = 0
        for image_path in examples["image"]:
            try:
                # 加载图像
                img = PILImage.open(image_path).convert("RGB")
                images.append(img)
            except Exception as e:
                error_count += 1
                logger.warning(f"无法加载图像 {image_path}: {e}")
                # 添加一个黑色图像作为替代
                img = PILImage.new("RGB", (224, 224), color=0)
                images.append(img)
        
        if error_count > 0:
            logger.warning(f"{split}集中有 {error_count} 个图像无法加载")
        
        examples["image"] = images
        return examples

    # 应用图像加载
    logger.info("加载训练集图像...")
    train_dataset = train_dataset.map(
        lambda examples: load_and_transform_images(examples, "train"),
        batched=True,
        batch_size=16,
        desc="加载训练集图像"
    )

    logger.info("加载测试集图像...")
    test_dataset = test_dataset.map(
        lambda examples: load_and_transform_images(examples, "test"),
        batched=True,
        batch_size=16,
        desc="加载测试集图像"
    )

    # 10. 设置特征信息
    train_dataset = train_dataset.cast_column("image", Image())
    test_dataset = test_dataset.cast_column("image", Image())

    # 创建ClassLabel对象
    class_labels = ClassLabel(num_classes=len(unique_labels), names=unique_labels)
    train_dataset = train_dataset.cast_column("label", class_labels)
    test_dataset = test_dataset.cast_column("label", class_labels)

    logger.info("数据集准备完成!")

    # 加载模型和处理器
    logger.info(f"加载模型: {args.model_name}")
    processor = AutoImageProcessor.from_pretrained(args.model_name)

    # 提取预处理参数
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    # 定义训练转换
    _train_transforms = Compose([
        Resize((size, size)),
        RandomRotation(90),
        RandomAdjustSharpness(2),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std)
    ])

    # 定义验证转换
    _val_transforms = Compose([
        Resize((size, size)),
        ToTensor(),
        Normalize(mean=image_mean, std=image_std)
    ])

    # 应用转换到数据集
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image) for image in examples['image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image) for image in examples['image']]
        return examples

    # 应用转换
    logger.info("应用数据转换...")
    train_dataset.set_transform(train_transforms)
    test_dataset.set_transform(val_transforms)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example['label'] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # 加载SigLIP模型用于分类
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
        seed=args.seed
    )

    # 初始化训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # 在训练前评估
    logger.info("训练前评估...")
    initial_results = trainer.evaluate()
    logger.info(f"训练前评估结果: {initial_results}")

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 训练后评估
    logger.info("训练后评估...")
    final_results = trainer.evaluate()
    logger.info(f"训练后评估结果: {final_results}")

    # 保存最终模型
    model_path = os.path.join(args.output_dir, "final")
    trainer.save_model(model_path)
    processor.save_pretrained(model_path)
    logger.info(f"模型已保存到 {model_path}")

    # 进行模型预测
    logger.info("生成预测...")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)

    # 生成分类报告
    y_true = test_df['label_id'].values
    y_pred = preds
    class_names = [id2label[i] for i in range(len(unique_labels))]

    logger.info("分类报告:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    logger.info("\n" + report)

    # 绘制混淆矩阵
    def plot_confusion_matrix(cm, classes, normalize=False, title='混淆矩阵', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            logger.info("归一化混淆矩阵")
        else:
            logger.info('未归一化混淆矩阵')

        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logger.info(f"混淆矩阵已保存到 {cm_path}")

    # 计算并绘制混淆矩阵
    logger.info("生成混淆矩阵...")
    confusion_mtx = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(confusion_mtx, classes=class_names, normalize=True, title='归一化混淆矩阵')

    logger.info("训练和评估完成！")

if __name__ == "__main__":
    main() 