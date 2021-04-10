import os
import re
import json
import time
import tensorflow as tf
import tokenization

from predict import process_one_example_p

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

vocab_file = "./vocab.txt"
tokenizer_ = tokenization.FullTokenizer(vocab_file=vocab_file)
label2id = json.loads(open("./label2id.json").read())
id2label = [k for k, v in label2id.items()]


def load_model(model_path):
    tf.reset_default_graph()
    output_graph_def = tf.GraphDef()
    with open(model_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")

    # tvars = tf.global_variables()
    # print('tvars', tvars)
    # for v in tvars:
    #     print('--initialized: %s, shape = %s' % (v.name, v.shape))
    #
    # exit()

    sess_ = tf.Session()
    sess_.run(tf.global_variables_initializer())

    return sess_


model_path = "pb/frozen_model.pb"
sess = load_model(model_path)
input_ids = sess.graph.get_tensor_by_name("input_ids:0")
input_mask = sess.graph.get_tensor_by_name("input_mask:0")  # is_training
segment_ids = sess.graph.get_tensor_by_name("segment_ids:0")  # fc/dense/Relu  cnn_block/Reshape
# keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
p = sess.graph.get_tensor_by_name("loss/ReverseSequence_1:0")


def predict(text):
    data = [text]
    # 逐个分成 最大62长度的 text 进行 batch 预测
    features = []
    for i in data:
        feature = process_one_example_p(tokenizer_, i, max_seq_len=64)
        features.append(feature)
    feed = {input_ids: [feature[0] for feature in features],
            input_mask: [feature[1] for feature in features],
            segment_ids: [feature[2] for feature in features],
            # keep_prob: 1.0
            }

    [probs] = sess.run([p], feed)
    result = []
    for index, prob in enumerate(probs):
        for v in prob[1:len(data[index]) + 1]:
            result.append(id2label[int(v)])
    # print(result)
    labels = {}
    start = None
    index = 0
    for w, t in zip("".join(data), result):
        if re.search("^[BS]", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label] = {te_: [[start, index - 1]]}
            start = index
            # print(start)
        if re.search("^O", t):
            if start is not None:
                # print(start)
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label] = {te_: [[start, index - 1]]}
            # else:
            #     print(start, labels)
            start = None
        index += 1
    if start is not None:
        # print(start)
        label = result[start][2:]
        if labels.get(label):
            te_ = text[start:index]
            # print(te_, labels)
            labels[label][te_] = [[start, index - 1]]
        else:
            te_ = text[start:index]
            # print(te_, labels)
            labels[label] = {te_: [[start, index - 1]]}
    # print(labels)
    return labels


def submit(path, save_name):
    data = []
    count = 0
    start_time = time.time()
    for line in open(path):
        if not line.strip():
            continue
        _ = json.loads(line.strip())
        res = predict(_["text"])
        count += 1
        data.append(json.dumps({"label": res}, ensure_ascii=False))
    used_time = time.time() - start_time
    print('count:', count)
    print('used_time:', used_time)
    print('inference/s:', count / used_time)
    print('one inference:', used_time / count)
    """
    CPU:
    count: 1343
    used_time: 65.40949296951294
    inference/s: 20.532187898566438
    one inference: 0.048704015613933685

    GPU:
    count: 1343
    used_time: 29.657947301864624
    inference/s: 45.28297209279768
    one inference: 0.022083356144351918
    """
    with open(save_name, "w") as f:
        f.write("\n".join(data))


if __name__ == '__main__':
    # 测试pb模型
    text = "梅塔利斯在乌克兰联赛、杯赛及联盟杯中保持9场不败，状态相当出色；"
    res_ = predict(text)
    print(res_)

    # submit("../cluener_public/dev.json", save_name='ner_predict_dev.json')
    # submit("../cluener_public/test.json", save_name='cluener_predict.json')

    """ GPU:
    count: 1343
    used_time: 29.807893753051758
    inference/s: 45.05517938054588
    one inference: 0.022195006517536676
    count: 1345
    used_time: 30.12023639678955
    inference/s: 44.65436400570085
    one inference: 0.022394227804304497
    """
