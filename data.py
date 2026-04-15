import random
import csv

# 正面词库
positive_subjects = ["手机", "电脑", "耳机", "屏幕", "系统", "电池", "物流", "客服", "包装", "做工"]
positive_verbs = ["很好", "不错", "流畅", "清晰", "给力", "满意", "喜欢", "优秀", "稳定", "顺畅"]
positive_adv = ["非常", "特别", "十分", "超级", "真的", "相当"]

# 负面词库
negative_subjects = ["手机", "电脑", "耳机", "屏幕", "系统", "电池", "物流", "客服", "包装", "做工"]
negative_verbs = ["很差", "卡顿", "失望", "一般", "糟糕", "难用", "垃圾", "严重", "崩溃", "不行"]
negative_adv = ["非常", "特别", "十分", "有点", "太", "真的"]

# 模板（更自然！🔥）
positive_templates = [
    "{sub}{adv}{verb}",
    "这个{sub}{adv}{verb}",
    "我觉得{sub}{adv}{verb}",
    "{sub}用起来{adv}{verb}",
    "整体来说{sub}{adv}{verb}"
]

negative_templates = [
    "{sub}{adv}{verb}",
    "这个{sub}{adv}{verb}",
    "我觉得{sub}{adv}{verb}",
    "{sub}用起来{adv}{verb}",
    "真的{sub}{adv}{verb}"
]

def generate_text(is_positive):
    if is_positive:
        template = random.choice(positive_templates)
        return template.format(
            sub=random.choice(positive_subjects),
            adv=random.choice(positive_adv),
            verb=random.choice(positive_verbs)
        )
    else:
        template = random.choice(negative_templates)
        return template.format(
            sub=random.choice(negative_subjects),
            adv=random.choice(negative_adv),
            verb=random.choice(negative_verbs)
        )

def generate_to_csv(file_path, total_size, chunk_size=10000):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'label'])

        for i in range(0, total_size, chunk_size):
            batch = []
            for _ in range(chunk_size):
                is_positive = random.choice([True, False])
                text = generate_text(is_positive)
                label = 1 if is_positive else 0
                batch.append([text, label])

            writer.writerows(batch)
            print(f"已生成 {i + chunk_size} 条数据")

# 🚀 一键生成 100万数据
generate_to_csv('data.csv', total_size=10_000)