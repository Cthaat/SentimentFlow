# 这些环境变量的目的，是让你不用改代码就能调训练参数：
    # - TRAIN_BATCH_SIZE：每次喂给 GPU 多少样本
    # - TRAIN_NUM_WORKERS：几个 CPU worker 同时读数据
    # - TRAIN_ACCUM_STEPS：梯度累积步数
    # - TRAIN_CHUNK_SIZE：每次从 CSV 读多少行
    # - TRAIN_DATASETS：多个 HF 数据集，逗号分隔
    #   例如：lansinuote/ChnSentiCorp,dataset2
    # - TRAIN_MAX_SAMPLES / TRAIN_MAX_VAL_SAMPLES：可选截断样本数