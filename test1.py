import torch
from torch.utils.data import IterableDataset
import pandas as pd

class Step5Detaset(IterableDataset):
    def __init__(self, csv_path, max_len=5):
        super().__init__()
        self.csv_path = csv_path
        self.max_len = max_len
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        
        df = pd.read_csv(self.csv_path)
        
        # 按 worker_id 分割数据，确保每个 worker 处理不同的数据，避免重复读取和竞争。
        for i, chunk in enumerate(pd.read_csv(self.csv_path, chunksize=3)):
            if i % num_workers != worker_id:
                continue
            chunk = chunk.sample(frac=1).reset_index(drop=True)  # 打乱顺序
            for _, row in chunk.iterrows():
                text = row['text']
                label = row['label']
                
                if not text.strip():
                    continue
                
                encoded = encode_text(text, max_len=self.max_len)
                yield (
                    torch.tensor(encoded, dtype=torch.long),
                    torch.tensor(label, dtype=torch.long)
                )

def encode_text(text, max_len=5):
    # 简单的编码示例：将每个字符转换为其 Unicode 码点，并截断或填充到固定长度。
    ids = [ord(c) for c in text]
    
    # 截断或填充到固定长度
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return ids

texts = ["hello world", "hi there", "a b", "single", " ", "", "  ", "ok", "no", "yes"]
labels = [5, 0, 4, 1, 3, 2, 0, 5, 1, 4]
ds = Step5Detaset("test.csv", max_len=5)

for x, y in ds:
    print(f"{x} -> {y}")
