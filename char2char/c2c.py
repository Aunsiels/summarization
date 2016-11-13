""" Char2Char implementation """
import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
from tensorflow.python.ops import array_ops
from pythonrouge import pythonrouge

VOCABULARY_SIZE = 256
EMBEDDING_SIZE_INPUT = 128
EMBEDDING_SIZE_OUTPUT = 512 #128 #512
BATCH_SIZE = 64 #5 # 64
MAX_LENGTH = 640 #200 # 640 #2048 # MULTIPLE OF STRIDE_POOLING !
FILTER_SIZES = [200, 250, 300, 300] #[20, 25, 30, 30] #
STRIDE_POOLING = 5
NUM_FILTERS = 2
HIGHWAY_LAYERS = 4
HIDDEN_SIZE_INPUT = 512 #128#512
HIDDEN_SIZE_OUTPUT = 1024 #128#1024

ROUGE_BASE = "/home/julien/Documents/programming/python/pythonrouge/pythonrouge/RELEASE-1.5.5/"
ROUGE_SCRIPT = ROUGE_BASE + "ROUGE-1.5.5.pl"
ROUGE_DATA = ROUGE_BASE + "data"

def get_rouge_score(peer_sentence, model_sentence):
    """get_rouge_score

    :param peer_sentence: Sentence generated
    :param model_sentence: True sentence
    """
    #'ROUGE-3', 'ROUGE-L', 'ROUGE-1', 'ROUGE-SU4', 'ROUGE-2'
    return pythonrouge.pythonrouge(peer_sentence, model_sentence, ROUGE_SCRIPT, ROUGE_DATA)

def get_sentence(f_open, max_length):
    """get_sentence Get a sentence from a file

    :param f_open:
    :param max_length:
    """
    #ADD begin/end of sentence
    line = f_open.readline()
    line = line[0:min(max_length - 2, len(line))]
    line = list(map(ord, line))
    if ord('\n') in line:
        line.remove(ord('\n'))
    if ord('\r') in line:
        line.remove(ord('\r'))
    line = [ord('<')] + line + [ord('>')] + [0] * (max_length - len(line) - 2)
    return line

#pylint: disable=too-many-instance-attributes
class Char2Char(object):
    """Char2Char"""

    #pylint: disable=too-many-arguments
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-statements
    def __init__(self, vocabulary_size, embedding_size_input,
                 max_length, filter_sizes, num_filters, stride_pooling,
                 n_highway_layers, embedding_size_output, hidden_size_input,
                 hidden_size_output):

        # INPUT and OUTPUT
        self.input_sentence = tf.placeholder(tf.int32, shape=[None, max_length], name="INPUT")
        self.output_sentence = tf.placeholder(tf.int32,
                                              shape=[None, max_length], name="OUTPUT")

        n_gru_encoder = int(math.ceil(max_length / stride_pooling))

        # Embeddings
        with tf.name_scope("embedding-input"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size_input], -0.01, 0.01),
                name="embedding-weigths-input")
            self.embed_input = tf.nn.embedding_lookup(embeddings,
                                                      self.input_sentence, name="embeddings-input")
            # We need 4D tensor for conv
            self.embed_input_expanded = tf.expand_dims(self.embed_input, -1)

        # Convolutions + maxPooling
        pooled_outputs = []
        for _, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution
                filter_shape = [filter_size, embedding_size_input, 1, num_filters]
                w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1, name="W"))
                b_conv = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embed_input_expanded,
                    w_conv,
                    strides=[1, 1, embedding_size_input, 1],
                    padding="SAME",
                    name="conv")
                # Non Linearity
                relu = tf.nn.relu(tf.nn.bias_add(conv, b_conv), name="relu")
                # Max pooling
                pooled = tf.nn.max_pool(
                    relu,
                    ksize=[1, stride_pooling, 1, 1],
                    strides=[1, stride_pooling, 1, 1],
                    padding="VALID",
                    name="pool")
                pooled_outputs.append(pooled)

        # Concatenate all convolutions + max pool
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        # Remove the useless dimension
        h_pool_flat = tf.reshape(self.h_pool, [-1, n_gru_encoder, num_filters_total])
        # Reduce dimension for multiplications
        h_pool_2d = tf.reshape(h_pool_flat, [-1, n_gru_encoder * num_filters_total])

        # Highway layers
        self.output_highway = [h_pool_2d]
        for i in range(1, n_highway_layers + 1):
            with tf.name_scope("Highway-layer-%d" % i):
                w_transform = tf.Variable(
                    tf.random_uniform([n_gru_encoder * num_filters_total,
                                       n_gru_encoder * num_filters_total],
                                      -0.01, 0.01),
                    name="w_transform")
                b_transform = tf.Variable(
                    tf.random_uniform([n_gru_encoder * num_filters_total], -0.01, 0.01),
                    name="b_transform")
                # Transform gate
                g_transform = tf.sigmoid(tf.matmul(self.output_highway[i - 1],
                                                   w_transform) + b_transform)
                # Carry gate
                g_carry = tf.ones([n_gru_encoder * num_filters_total], name="ones") - g_transform
                w_linear = tf.Variable(
                    tf.random_uniform([n_gru_encoder * num_filters_total,
                                       n_gru_encoder * num_filters_total], -0.01, 0.01),
                    name="w_linear")
                b_linear = tf.Variable(
                    tf.random_uniform([n_gru_encoder * num_filters_total], -0.01, 0.01),
                    name="b_linear")
                relu = tf.nn.relu(tf.matmul(self.output_highway[i - 1], w_linear) + b_linear)
                output = tf.mul(g_transform, relu) + tf.mul(g_carry, self.output_highway[i - 1])
                self.output_highway.append(output)

        highway_reshape = tf.reshape(self.output_highway[-1],
                                     [-1, n_gru_encoder, num_filters_total])

        self.h_pool_trans = tf.transpose(highway_reshape, [1, 0, 2])
        self.h_pool_reshape = tf.reshape(self.h_pool_trans, [-1, num_filters_total])
        self.h_pool_split = tf.split(0, n_gru_encoder, self.h_pool_reshape)

        print("**POOLING***", self.h_pool_split[0].get_shape())
        print("LENGTH", len(self.h_pool_split))

        #GRU
        with tf.name_scope("BIGRU-Input"):
            gru_forward = rnn_cell.GRUCell(hidden_size_input)
            gru_backward = rnn_cell.GRUCell(hidden_size_input)
            self.outputs_gru_input, _, _ = rnn.bidirectional_rnn(gru_forward, gru_backward,
                                                                 self.h_pool_split,
                                                                 dtype=tf.float32)

        print("**OUTGRU****", self.outputs_gru_input[0].get_shape())
        print("LENGTH", len(self.outputs_gru_input))


        batch_size = array_ops.shape(self.h_pool_split[0])[0]

        # Embeddings output
        with tf.name_scope("embedding-output"):
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size_output], -0.01, 0.01),
                name="embedding-weigths-input")
            zeros = tf.zeros([batch_size, 1], dtype=tf.int32)
            output_prepend = tf.concat(1, [zeros, self.output_sentence])
            output_slice = tf.slice(output_prepend, [0, 0], [batch_size, max_length])
            self.embed_output = tf.nn.embedding_lookup(embeddings,
                                                       output_slice,
                                                       name="embeddings-input")

        self.embed_trans = tf.transpose(self.embed_output, [1, 0, 2])
        self.embed_reshape = tf.reshape(self.embed_trans, [-1, embedding_size_output])
        self.embed_split = tf.split(0, max_length, self.embed_reshape)

        print("****EMBED*****", self.embed_split[0].get_shape())
        print("LENGTH", len(self.embed_split))

        #GRU
        with tf.name_scope("first-layer-decoder"):
            gru_forward = rnn_cell.GRUCell(hidden_size_output)
            self.outputs_first_layer, _ = rnn.rnn(gru_forward,
                                                  self.outputs_gru_input,
                                                  dtype=tf.float32)

        with tf.name_scope("attention-decoder"):
            gru_decoder = rnn_cell.GRUCell(hidden_size_output)
            self.initial_state = gru_decoder.zero_state(batch_size, tf.float32)
            top_states = [array_ops.reshape(e, [-1, 1, hidden_size_output])
                          for e in self.outputs_first_layer]
            self.attention_states = array_ops.concat(1, top_states)
            self.output_decoder, _ = seq2seq.attention_decoder(self.embed_split,
                                                               self.initial_state,
                                                               self.attention_states,
                                                               gru_decoder)

        print("*OUTPUTDEC**", self.output_decoder[0].get_shape())
        print("LENGTH", len(self.output_decoder))

        w_end = tf.Variable(
            tf.random_uniform([hidden_size_output, vocabulary_size],
                              -0.01, 0.01))

        self.decoder_lin = []
        for output in self.output_decoder:
            self.decoder_lin.append(tf.batch_matmul(output, w_end))

        output_split = tf.split(1, max_length, self.output_sentence)

        print("*OUTPUTDES**", self.decoder_lin[0].get_shape())
        print("LENGTH", len(self.decoder_lin))

        self.output_decoder_softmax = []
        for output in self.decoder_lin:
            self.output_decoder_softmax.append(tf.nn.softmax(
                output))

        with tf.name_scope("loss"):
            weights = []
            for i in range(max_length):
                weights.append(tf.ones([batch_size],
                                       name="weight-%d"%i))

            self.loss = seq2seq.sequence_loss(self.decoder_lin,
                                              output_split,
                                              weights)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
                                                beta1=0.9,
                                                beta2=0.999,
                                                epsilon=1e-08,
                                                use_locking=False,
                                                name='Adam').minimize(self.loss)

        self.sess = tf.Session()
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)
        self.summary_writer = tf.train.SummaryWriter('/tmp', graph=self.sess.graph)

        self.saver = tf.train.Saver()
        #self.load_model("model_test")

        print("Graph Built")

    def save_model(self, filename):
        """saveModel Save the model

        :param filename:
        """
        self.saver.save(self.sess, filename)

    def load_model(self, filename):
        """load_model Load model

        :param filename:
        """
        self.saver.restore(self.sess, filename)

    def generate_example(self, batch_size, max_length, vocabulary_size):
        """generate_example
            Generate example to train the network. This is simply a sentence
            as an input and the same reversed sentence as output

        :param batch_size: The size of the batch
        :param max_length: The length of the sentences to generate
        :param vocabulary_size: The number of tokens in the vocabulary
        """
        input_x = []
        input_y = []
        for _ in range(batch_size):
            temp_x = [random.randint(0, vocabulary_size - 1) for j in range(max_length)]
            temp_y = temp_x.copy()
            temp_y.reverse()
            input_x.append(temp_x)
            input_y.append(temp_y)
        return {self.input_sentence : input_x, self.output_sentence : input_y}

    def generate_batch(self, f_open_input, f_open_output, max_length, batch_size):
        """generate_batch Get a batch of sentences

        :param f_open_input:
        :param f_open_output:
        :param max_length:
        :param batch_size:
        """
        res_input = []
        res_output = []
        for _ in range(batch_size):
            temp_input = get_sentence(f_open_input, max_length)
            if temp_input[0] == 0:
                return None
            res_input.append(temp_input)
            res_output.append(get_sentence(f_open_output, max_length))
        return {self.input_sentence : res_input, self.output_sentence : res_output}

    def test_train(self, steps):
        """test_train
            Train the network on reverse tack

        :param steps: Number of training steps
        """
        for step in range(steps):
            feed_dict = self.generate_example(BATCH_SIZE, MAX_LENGTH,
                                              VOCABULARY_SIZE)
            result, _, loss_value = self.sess.run([self.output_decoder_softmax,
                                                   self.optimizer,
                                                   self.loss],
                                                  feed_dict=feed_dict)
            if step % 1 == 0:
                print('Step %d: loss = %.2f' % (step, loss_value))
                print(result[0].shape)
                print(" ".join([str(x) for x in feed_dict[self.input_sentence][0]]))
                print(" ".join([str(np.argmax(result[i][0])) for i in range(MAX_LENGTH)]))

    def train_epoch(self, name_article, name_title, max_length, batch_size):
        """train_epoch Train over a full epoch

        :param name_article:
        :param name_title:
        :param max_length:
        :param batch_size:
        """
        f_input = open(name_article)
        f_output = open(name_title)
        batch = self.generate_batch(f_input, f_output, max_length, batch_size)
        step = 0
        rouge_1 = []
        rouge_2 = []
        rouge_3 = []
        rouge_su4 = []
        rouge_l = []
        loss_total = []
        while batch is not None:
            if step % 1000 == 0:
                result, _, loss_value = self.sess.run([self.output_decoder_softmax,
                                                       self.optimizer,
                                                       self.loss],
                                                      feed_dict=batch)
                print('Step %d: loss = %.8f' % (step, loss_value))
                print(result[0].shape)
                print("".join([chr(x) for x in batch[self.input_sentence][0]]))
                print("".join([chr(x) for x in batch[self.output_sentence][0]]))
                print("".join([chr(np.argmax(result[i][0])) for i in range(max_length)]))
                rouge_1_temp = 0
                rouge_2_temp = 0
                rouge_3_temp = 0
                rouge_su4_temp = 0
                rouge_l_temp = 0
                loss_temp = loss_value
                for i in range(batch_size):
                    temp = get_rouge_score("".join([chr(np.argmax(result[j][i]))
                                                    for j in range(max_length)]),
                                           "".join([chr(x)
                                                    for x in
                                                    batch[self.output_sentence][i]]))
                    rouge_1_temp += temp["ROUGE-1"]
                    rouge_2_temp += temp["ROUGE-2"]
                    rouge_3_temp += temp["ROUGE-3"]
                    rouge_l_temp += temp["ROUGE-L"]
                    rouge_su4_temp += temp["ROUGE-SU4"]
                rouge_1.append(str(rouge_1_temp))
                rouge_2.append(str(rouge_2_temp))
                rouge_3.append(str(rouge_3_temp))
                rouge_l.append(str(rouge_l_temp))
                rouge_su4.append(str(rouge_su4_temp))
                loss_total.append(str(loss_temp))
            else:
                _, loss_value = self.sess.run([self.optimizer,
                                               self.loss],
                                              feed_dict=batch)
            if step%100000 == 0:
                self.save_model("model_test")
                f_register = open("loss.csv", "a")
                f_register.write(",".join(rouge_1))
                f_register.write("\n")
                f_register.write(",".join(rouge_2))
                f_register.write("\n")
                f_register.write(",".join(rouge_3))
                f_register.write("\n")
                f_register.write(",".join(rouge_l))
                f_register.write("\n")
                f_register.write(",".join(rouge_su4))
                f_register.write("\n")
                f_register.write(",".join(loss_total))
                f_register.write("\n")
                rouge_1 = []
                rouge_2 = []
                rouge_3 = []
                rouge_su4 = []
                rouge_l = []
                loss_total = []
                f_register.close()
            step += 1
            batch = self.generate_batch(f_input, f_output, max_length, batch_size)
        f_input.close()
        f_output.close()

    def validation(self, name_article, name_title, max_length, batch_size):
        """validation

        :param name_article: The name of the file containing the text of
        the article
        :param name_title: The name of the file containing the titles of
        the article
        :param max_length: The maximum length of a sentence
        :param batch_size: The size of a batch
        """
        f_input = open(name_article)
        f_output = open(name_title)
        batch = self.generate_batch(f_input, f_output, max_length, batch_size)
        count_size = 0
        sum_score = 0
        sum_loss = 0
        while batch is not None:
            result, loss_value = self.sess.run([self.output_decoder_softmax,
                                                self.loss],
                                               feed_dict=batch)
            for i in range(batch_size):
                count_size += 1
                sum_score += get_rouge_score("".join([chr(np.argmax(result[j][i]))
                                                      for j in range(max_length)]),
                                             "".join([chr(x)
                                                      for x in
                                                      batch[self.output_sentence][i]]))['ROUGE-1']
                sum_loss += loss_value
            batch = self.generate_batch(f_input, f_output, max_length, batch_size)
        f_input.close()
        f_output.close()
        return sum_score / count_size, sum_loss / count_size

    def evaluation(self, name_article, name_title, max_length, batch_size):
        """evaluation

        :param name_article: The name of the file containing the text of
        the article
        :param name_title: The name of the file containing the titles of
        the article
        :param max_length: The maximum length of a sentence
        :param batch_size: The size of a batch
        """
        f_input = open(name_article)
        f_output = open(name_title)
        batch = self.generate_batch(f_input, f_output, max_length, batch_size)
        count_size = 0
        sum_loss = 0
        score_r1 = 0
        score_r2 = 0
        score_r3 = 0
        score_l = 0
        score_su4 = 0

    #'ROUGE-3', 'ROUGE-L', 'ROUGE-1', 'ROUGE-SU4', 'ROUGE-2'
        while batch is not None:
            result, loss_value = self.sess.run([self.output_decoder_softmax,
                                                self.loss],
                                               feed_dict=batch)
            for i in range(batch_size):
                count_size += 1
                score = get_rouge_score("".join([chr(np.argmax(result[j][i]))
                                                 for j in range(max_length)]),
                                        "".join([chr(x)
                                                 for x in
                                                 batch[self.output_sentence][i]]))
                score_r1 += score["ROUGE-1"]
                score_r2 += score["ROUGE-2"]
                score_r3 += score["ROUGE-3"]
                score_l += score["ROUGE-L"]
                score_su4 += score["ROUGE-SU4"]
                sum_loss += loss_value
            batch = self.generate_batch(f_input, f_output, max_length, batch_size)
        f_input.close()
        f_output.close()
        score_r1 /= count_size
        score_r2 /= count_size
        score_r3 /= count_size
        score_l /= count_size
        score_su4 /= count_size
        sum_loss /= count_size
        print("EVALUATION SCORE : ")
        print("ROUGE 1 : %f"%score_r1)
        print("ROUGE 2 : %f"%score_r2)
        print("ROUGE 3 : %f"%score_r3)
        print("ROUGE L : %f"%score_l)
        print("ROUGE SU4 : %f"%score_su4)
        print("LOSS : %f"%sum_loss)

C2C = Char2Char(VOCABULARY_SIZE, EMBEDDING_SIZE_INPUT,
                MAX_LENGTH, FILTER_SIZES, NUM_FILTERS, STRIDE_POOLING,
                HIGHWAY_LAYERS, EMBEDDING_SIZE_OUTPUT, HIDDEN_SIZE_INPUT,
                HIDDEN_SIZE_OUTPUT)

BESTR1 = 0.0
NEXTR1 = 0.0

while NEXTR1 >= BESTR1:
    BESTR1 = NEXTR1
    C2C.train_epoch("/run/media/julien/MyPassport/LDC/train.article.txt",
                    "/run/media/julien/MyPassport/LDC/train.title.txt",
                    MAX_LENGTH,
                    BATCH_SIZE)
    R1SCORE, LSCORE = C2C.validation("/run/media/julien/MyPassport/LDC/valid.article.txt",
                                     "/run/media/julien/MyPassport/LDC/valid.title.txt",
                                     MAX_LENGTH,
                                     BATCH_SIZE)
    print("Validation Score : R1:%f, Loss:%f"%(R1SCORE, LSCORE))
    NEXTR1 = R1SCORE

C2C.evaluation("/run/media/julien/MyPassport/LDC/test.article.txt",
               "/run/media/julien/MyPassport/LDC/test.title.txt",
               MAX_LENGTH,
               BATCH_SIZE)
