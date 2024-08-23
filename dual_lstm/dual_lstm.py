import  tensorflow as tf
from collections import  namedtuple
from tensorflow.contrib import rnn
# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  91620,
  "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 256, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 16, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "rnn_dim",
    "vocab_size",
    "glove_path",
    "vocab_path"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim)


def model_imp(ques, ques_len, ans, ans_len, ans_f, ans_f_len, batch_size):
    n_characters = 2
    hparams = create_hparams()
    w_embed = tf.get_variable('w_embed', shape=[hparams.vocab_size, n_characters * hparams.embedding_dim],
                              initializer=tf.random_uniform_initializer(-1.0, 1.0))
    w_embed_2 = tf.get_variable('w_embed_2', shape=[hparams.vocab_size, n_characters * hparams.embedding_dim],
                                initializer=tf.random_uniform_initializer(-1.0, 1.0))

    # 1.2 --- rnn for question ---
    ques_1 = tf.nn.embedding_lookup(w_embed, ques, name='ques_1')
    ques_2 = tf.nn.embedding_lookup(w_embed_2, ques, name='ques_2')

    # 1.2.0 --- calculate the distribution for the question ---
    with tf.variable_scope('character') as vs_latent_character:
        cell = rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        cell_r = rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        output, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, tf.concat(2, [ques_1, ques_2]),
                                                        sequence_length=ques_len, dtype=tf.float32)
        character_information = tf.reduce_max(tf.concat(2, [output[0], output[1]]), 1)
        T = tf.get_variable('T', shape=[hparams.rnn_dim * 2, n_characters])
        character_dist = tf.expand_dims(tf.nn.softmax(tf.matmul(character_information, T), -1), 1)
        # character = tf.argmax(tf.matmul(character_information, T), 1)
        # character_dist = tf.expand_dims(tf.one_hot(character, n_characters, on_value = 1.0, off_value = 0.0), 1)
        print (character_dist.get_shape())

    # 1.2.1 -- Three different ques-ans combinations ---
    with tf.variable_scope('rnn_ques') as vs_ques:
        cell = rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        cell_r = rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        output_ques, state_ques = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, ques_1, sequence_length=ques_len,
                                                                  dtype=tf.float32)

    with tf.variable_scope('rnn_ques2') as vs_ques:
        cell_2 = rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        cell_r_2 = rnn.LSTMCell(hparams.rnn_dim, forget_bias=2.0, use_peepholes=True, state_is_tuple=True)
        output_ques_2, state_ques = tf.nn.bidirectional_dynamic_rnn(cell_2, cell_r_2, ques_2, sequence_length=ques_len,
                                                                    dtype=tf.float32)

    ques_output_1 = tf.reduce_max(tf.concat(2, [output_ques[0], output_ques[1]]), 1)
    ques_output_2 = tf.reduce_max(tf.concat(2, [output_ques_2[0], output_ques_2[1]]), 1)
    ques_output = tf.batch_matmul(character_dist, tf.pack([ques_output_1, ques_output_2], axis=1))
    ques_output = tf.squeeze(ques_output, [1])

    M = tf.get_variable('M', shape=[hparams.rnn_dim * 2, hparams.rnn_dim * 2],
                        initializer=tf.random_uniform_initializer(-1.0, 1.0))
    ques_output = tf.matmul(ques_output, M)

    # 1.3 --- rnn for ans ---
    ans_1 = tf.nn.embedding_lookup(w_embed, ans, name='ans')
    ans_f_1 = tf.nn.embedding_lookup(w_embed, ans_f, name='ans_f')
    ans_2 = tf.nn.embedding_lookup(w_embed_2, ans, name='ans')
    ans_f_2 = tf.nn.embedding_lookup(w_embed_2, ans_f, name='ans_f')
    with tf.variable_scope('rnn_ques', reuse=True) as vs_ans:
        output_1, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, ans_1, sequence_length=ans_len,
                                                          dtype=tf.float32)
        output_f1, state = tf.nn.bidirectional_dynamic_rnn(cell, cell_r, ans_f_1, sequence_length=ans_f_len,
                                                           dtype=tf.float32)
    with tf.variable_scope('rnn_ques2', reuse=True) as vs_ans:
        output_2, state = tf.nn.bidirectional_dynamic_rnn(cell_2, cell_r_2, ans_2, sequence_length=ans_len,
                                                          dtype=tf.float32)
        output_f2, state = tf.nn.bidirectional_dynamic_rnn(cell_2, cell_r_2, ans_f_2, sequence_length=ans_f_len,
                                                           dtype=tf.float32)

    ans_output_1 = tf.reduce_max(tf.concat(2, [output_1[0], output_1[1]]), 1)
    ans_output_2 = tf.reduce_max(tf.concat(2, [output_2[0], output_2[1]]), 1)

    ans_output = tf.batch_matmul(character_dist, tf.pack([ans_output_1, ans_output_2], axis=1))
    ans_output = tf.squeeze(ans_output, [1])

    ans_output_f1 = tf.reduce_max(tf.concat(2, [output_f1[0], output_f1[1]]), 1)
    ans_output_f2 = tf.reduce_max(tf.concat(2, [output_f2[0], output_f2[1]]), 1)

    ans_output_f = tf.batch_matmul(character_dist, tf.pack([ans_output_f1, ans_output_f2], axis=1))
    ans_output_f = tf.squeeze(ans_output_f, [1])

    # 1.4 -----------------    the prediction part ---------------------------

    ques_output = tf.nn.l2_normalize(ques_output, 1)
    ans_output = tf.nn.l2_normalize(ans_output, 1)
    ans_output_f = tf.nn.l2_normalize(ans_output_f, 1)

    prob = [ques_output, ans_output]
    simi = tf.reduce_sum(tf.mul(ques_output, ans_output), 1)
    simi_f = tf.reduce_sum(tf.mul(ques_output, ans_output_f), 1)

    loss = tf.maximum(0.0, 0.25 - simi + simi_f)

    loss_ = tf.reduce_mean(loss)
    return prob, loss_

def main():
    hparams = create_hparams()

    # model_fn =
    #
    # estimator = tf.contrib.learn.Estimator(
    #     model_fn=model_fn,
    #     model_dir=MODEL_DIR,
    #     config=tf.contrib.learn.RunConfig())
    #
    # input_fn_train = udc_inputs.create_input_fn(
    #     mode=tf.contrib.learn.ModeKeys.TRAIN,
    #     input_files=[TRAIN_FILE],
    #     batch_size=hparams.batch_size,
    #     num_epochs=FLAGS.num_epochs)
    #
    # input_fn_eval = udc_inputs.create_input_fn(
    #     mode=tf.contrib.learn.ModeKeys.EVAL,
    #     input_files=[VALIDATION_FILE],
    #     batch_size=hparams.eval_batch_size,
    #     num_epochs=1)
    #
    # eval_metrics = udc_metrics.create_evaluation_metrics()
    #
    # eval_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    #     input_fn=input_fn_eval,
    #     every_n_steps=FLAGS.eval_every,
    #     metrics=eval_metrics)
    #
    # estimator.fit(input_fn=input_fn_train, steps=None, monitors=[eval_monitor])


if __name__ == '__main__':
     print('00')
