def reindex_sequences(seq_samples, src_word_index, dst_word_index):
    src_index_word = {v: k for k, v in src_word_index.items()}
    for seq in seq_samples:
        for i in range(len(seq)):
            idx = seq[i]
            word = src_index_word[idx]
            seq[i] = dst_word_index[word]


def check_mode(mode):
    if mode not in ['categorical', 'binary', 'ensemble']:
        raise ValueError("mode argument must be 'categorical', 'binary' or 'ensemble'")