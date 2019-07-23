import os
import sys
import argparse
import cv2

import tensorflow as tf
import numpy as np

from functools import partial
from math import sqrt, ceil

from enum import Enum


class NetType(Enum):
    LSTM = 1
    CONV = 2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

characters = [' ', '!', '"', '&', '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
              ':', ';', '=', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
              'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '}', '~', '°',
              'é', '§', '$', '+', '%', "'", '©', '|', '\\', '#', '@']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", help="Path to directory of images to be transcribed.", required=True)
    parser.add_argument("--net", help="Parameter for choosing the network. Use LSTM (default) or CONV.", required=False,
                        default="LSTM")
    return parser.parse_args()


def sequential_conv_block(net, train_phase, layer_count, filter_count, kernel_size=3, dilations=[1, 1], padding='same'):
    for i in range(layer_count):
        net = tf.layers.conv2d(
            net, filters=filter_count, kernel_size=kernel_size,
            dilation_rate=dilations, padding=padding, use_bias=False)
        net = tf.layers.batch_normalization(net, training=train_phase, fused=True)
        net = tf.nn.relu(net)
    return net


def sequential_conv_block_1d(net, train_phase, layer_count, filter_count, kernel_size=3, dilations=1, padding='same'):
    for i in range(layer_count):
        net = tf.layers.conv1d(
            net, filters=filter_count, kernel_size=kernel_size,
            dilation_rate=dilations, padding=padding)
        net = tf.layers.batch_normalization(net, training=train_phase, fused=True)
        net = tf.nn.relu(net)
    return net


def build_eval_net(data_shape, class_count, net_builder):
    keep_prob = 1
    train_phase = False
    input_data = tf.placeholder(tf.uint8, shape=data_shape, name='input_data')
    seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

    net = tf.cast(input_data, tf.float32) / 255

    transformed_data = net

    decoded, logits, logits_t, log_prob = net_builder(class_count, net, train_phase, keep_prob, seq_len)

    saver = tf.train.Saver()

    return saver, input_data, transformed_data, seq_len, logits, logits_t, decoded, log_prob


def get_aggregation(net, count, train_phase):
    height = int(net.shape[1])
    net = sequential_conv_block(net, train_phase, 1, count, kernel_size=(height, 1), padding="valid")
    net = net[:, 0, :, :]
    return net


def build_deep_net(class_count, input_tensor, train_phase, keep_prob, seq_len,
                   num_hidden=96, num_recurrent_layers=1, block_layer_count=2, base_filter_count=12, block_count=3,
                   output_subsampling=4):
    class_count = class_count + 1
    net = input_tensor

    bypass = []
    for i in range(block_count):
        net = sequential_conv_block(net, train_phase, block_layer_count, base_filter_count * (2 ** i))
        net = tf.layers.max_pooling2d(net, 2, 2)
        net = tf.nn.dropout(net, keep_prob)
        bp = get_aggregation(net, base_filter_count * 2, train_phase)
        if i != block_count - 1:
            bp = tf.layers.max_pooling1d(bp, 2 ** (block_count - i - 1), 2 ** (block_count - i - 1))
        bypass.append(bp)

    net = sequential_conv_block(net, train_phase, 1, num_hidden, padding='same')

    net = get_aggregation(net, num_hidden, train_phase)
    net = tf.nn.dropout(net, keep_prob ** 2)
    bypass.append(net)

    net = tf.concat(bypass + [net], axis=2)

    if num_recurrent_layers > 0:
        lstm = tf.contrib.cudnn_rnn.CudnnGRU(num_recurrent_layers, num_hidden, direction='bidirectional', name='awesome_lstm')
        net = tf.transpose(net, [1, 0, 2])
        net, x = lstm(net)
        net = tf.transpose(net, [1, 0, 2])

    net = tf.layers.batch_normalization(net, training=train_phase)
    net = tf.nn.relu(net)
    net = tf.nn.dropout(net, keep_prob ** 2)

    net = tf.concat(bypass + [net], axis=2)

    for i in range(block_count - int(sqrt(output_subsampling))):
        net = tf.keras.layers.UpSampling1D()(net)
        net = sequential_conv_block_1d(net, train_phase, block_layer_count, num_hidden)

    logits = tf.layers.conv1d(net, class_count, 1)

    logits_t = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits_t, seq_len, merge_repeated=True)

    return decoded, logits, logits_t, log_prob


def define_net(net_type: NetType):
    if net_type == NetType.LSTM:
        return partial(
            build_deep_net, num_hidden=96, num_recurrent_layers=1,
            block_layer_count=2, base_filter_count=12, block_count=3, output_subsampling=4)
    else:
        return partial(
            build_deep_net, num_hidden=96, num_recurrent_layers=0,
            block_layer_count=2, base_filter_count=12, block_count=3, output_subsampling=4)


def load_net(net_type: NetType):
    batch_size = 1
    line_px_height = 48
    data_shape = [batch_size, line_px_height, None, 3]

    net_graph = tf.Graph()
    tf.reset_default_graph()
    with net_graph.as_default():
        net = define_net(net_type)
        (saver, input_data, _, seq_len, logits, logits_t, decoded, _) = build_eval_net(
            [batch_size, line_px_height, None, 3], len(characters), net)

    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True

    session = tf.Session(graph=net_graph, config=config)

    if net_type == NetType.LSTM:
        checkpoint = "lstm_net/cnn_lstm_ctc.ckpt"
    else:
        checkpoint = "conv_net/cnn_ctc.ckpt"

    saver.restore(session, checkpoint)

    tmp_data_shape = [batch_size, line_px_height, 128, 3]

    out_logits, = session.run([logits], feed_dict={input_data: np.zeros(tmp_data_shape, dtype=np.uint8)})
    net_subsampling = tmp_data_shape[2] / out_logits.shape[1]

    return saver, input_data, seq_len, logits, logits_t, decoded, net_subsampling, data_shape, config, session


def pad_image(image):
    image_width = image.shape[1]

    padded_image_width = int(ceil(image_width/32.0) * 32)
    padded_image_width += 1024

    image_start = int((padded_image_width - image_width) / 2)
    image_end = image_start + image_width

    padded_image_shape = (image.shape[0], padded_image_width, image.shape[2])
    padded_image = np.zeros(padded_image_shape)

    padded_image[:, image_start:image_end] = image

    return padded_image


def transcribe(net, image):
    saver, input_data, seq_len, logits, logits_t, decoded, net_subsampling, data_shape, config, session = net

    padded_image = pad_image(image)
    data = np.expand_dims(padded_image, axis=0)

    seq_lengths = np.ones([data_shape[0]], dtype=np.int32) * data.shape[2] / net_subsampling

    out_decoded, out_logits = session.run(
        [decoded, logits],
        feed_dict={input_data: data, seq_len: seq_lengths})

    transcription = None

    pos, = np.nonzero(out_decoded[0].indices[:, 0] == 0)
    if pos.size:
        transcription = ""
        for val in out_decoded[0].values[pos]:
            transcription += characters[val]

    return transcription, padded_image


def process(input_dir, network):
    network = network.lower()

    if network == "lstm":
        net_type = NetType.LSTM
    elif network == "conv":
        net_type = NetType.CONV
    else:
        print("Unknown network '{network}'. Please use one of these options: LSTM, CONV.".format(network=network))
        return

    net = load_net(net_type)
    files = [file for file in os.listdir(input_dir) if file.lower().endswith(('.jpg', '.png'))]

    for file in files:
        image = cv2.imread(os.path.join(input_dir, file))
        transcription, padded_image = transcribe(net, image)

        cv2.imwrite(os.path.join(input_dir, "padded", file), padded_image)

        print("{filename} {transcription}".format(filename=file, transcription=transcription))


def main():
    args = parse_args()

    process(args.input_dir, args.net)

    return 0


if __name__ == "__main__":
    sys.exit(main())
