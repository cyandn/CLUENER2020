punctuations = ['；', ';', '！', '!', '？', '?', '。', '.', '…']


def is_punctuation_in(text):
    for item in punctuations:
        if item in text:
            return True
    return False


def split_seq(x, seq_length, y=None):
    """
    seq_length: 减去 [CLS] 和 [SEP] 的长度
    """
    if y is not None:
        assert len(x) == len(y)

    temp_x = []
    temp_y = []

    start_idx = 0
    while start_idx < len(x):
        temp_seq_x = x[start_idx:start_idx + seq_length]
        temp_seq_len = len(temp_seq_x)
        if temp_seq_len == seq_length and is_punctuation_in(temp_seq_x):
            end_idx = temp_seq_len - 1
            while end_idx >= 0:
                if temp_seq_x[end_idx] in punctuations:
                    temp_x.append(x[start_idx:start_idx + end_idx + 1])
                    if y is not None:
                        temp_y.append(y[start_idx:start_idx + end_idx + 1])
                    break
                end_idx = end_idx - 1
            start_idx = start_idx + end_idx + 1
        else:
            temp_x.append(temp_seq_x)
            if y is not None:
                temp_y.append(y[start_idx:start_idx + seq_length])
            start_idx = start_idx + temp_seq_len

    for xx, yy in zip(temp_x, temp_y):
        assert len(xx) == len(yy)

    if y is not None:
        return temp_x, temp_y
    else:
        return temp_x
