"""诊断脚本：检查数据集标签有效性。"""

import os
from datasets import load_dataset

def check_dataset_labels(name: str):
    """检查数据集的标签范围。"""
    print(f"\n{'='*60}")
    print(f"Checking dataset: {name}")
    print('='*60)
    
    try:
        ds = load_dataset(name)
        
        # 获取 train split
        if "train" not in ds:
            print(f"⚠️  No train split found")
            return False
        
        train_split = ds["train"]
        print(f"Train samples: {len(train_split)}")
        
        # 检查标签列
        if "label" not in train_split.column_names:
            print(f"❌ No 'label' column found. Available columns: {train_split.column_names}")
            return False
        
        labels = train_split["label"]
        
        # 统计标签值
        unique_labels = set()
        label_counts = {}
        invalid_labels = []
        
        for idx, label in enumerate(labels):
            label_val = int(label) if label is not None else None
            
            if label_val is None:
                invalid_labels.append((idx, label, "None value"))
            elif label_val < 0 or label_val > 1:
                invalid_labels.append((idx, label, f"Out of range [0,1]"))
            else:
                unique_labels.add(label_val)
                label_counts[label_val] = label_counts.get(label_val, 0) + 1
        
        print(f"\n✅ Unique valid labels: {sorted(unique_labels)}")
        print(f"Label distribution:")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            print(f"  {label}: {count} samples")
        
        if invalid_labels:
            print(f"\n⚠️  Found {len(invalid_labels)} invalid labels!")
            for idx, val, reason in invalid_labels[:10]:  # 只显示前10个
                print(f"  Index {idx}: {val!r} ({reason})")
            if len(invalid_labels) > 10:
                print(f"  ... and {len(invalid_labels) - 10} more")
            return False
        else:
            print(f"\n✅ All labels are valid!")
            return True
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

if __name__ == "__main__":
    dataset_names = [
        item.strip()
        for item in os.getenv("TRAIN_DATASETS", "lansinuote/ChnSentiCorp").split(",")
        if item.strip()
    ]
    
    print("Checking datasets...")
    results = {}
    for name in dataset_names:
        results[name] = check_dataset_labels(name)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print('='*60)
    for name, valid in results.items():
        status = "✅ Valid" if valid else "❌ Invalid"
        print(f"{status}: {name}")
