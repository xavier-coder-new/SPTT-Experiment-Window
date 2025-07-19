global_param = {
    'time_for_batch': 0.0,
    'min_time': 1e6,
    'max_time': 0.0,
    'flag': False,
    'compression_finished': False,
    'time_block_size': 0.0,
    'total_time_block_size': 0.0,
    # 用于Dropout估计
    'trials': 0,
    'dropout_x': 0.0,
    'dropout_d': 0.0,
    # 用于Hutchinson估计
    'hutchinson_size': 0,
    # 用于随机采样估计
    'random_size': 0,
    'chunk_index': 0
}