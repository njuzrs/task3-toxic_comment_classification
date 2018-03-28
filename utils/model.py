# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 20:47:19 2018

@author: zrssch
"""
import tensorflow as tf

def model_fn(features, targets, mode, params):
    embedding_matrix = tf.convert_to_tensor(params["embedding_matrix"], dtype=tf.float32)
    print("embedding: ",embedding_matrix)
    embedding_layer = tf.nn.embedding_lookup(embedding_matrix, features)
    print("embedding layer: ",embedding_layer)
    embedding_layer = tf.transpose(embedding_layer, [1, 0, 2])
    steps = int(embedding_layer.shape[0])
    sizes = int(embedding_layer.shape[2])
    embedding_layer = tf.reshape(embedding_layer, [-1, sizes])
    embedding_layer = tf.split(embedding_layer, steps)
    gru_fw_cell_1 = tf.nn.rnn_cell.GRUCell(params["recurrent_units"])
    gru_fw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell_1, output_keep_prob=1-params["dropout_rate"])
    gru_fw_cell_2 = tf.nn.rnn_cell.GRUCell(params["recurrent_units"])
    gru_fw_cell_2 = tf.contrib.rnn.AttentionCellWrapper(gru_fw_cell_2, attn_length=40)
    gru_bw_cell_1 = tf.nn.rnn_cell.GRUCell(params["recurrent_units"])
    gru_bw_cell_1 = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell_1, output_keep_prob=1-params["dropout_rate"])
    gru_bw_cell_2 = tf.nn.rnn_cell.GRUCell(params["recurrent_units"])
    gru_bw_cell_2 = tf.contrib.rnn.AttentionCellWrapper(gru_bw_cell_2, attn_length=40)
    x = tf.contrib.rnn.stack_bidirectional_rnn([gru_fw_cell_1,gru_fw_cell_2], [gru_bw_cell_1,gru_bw_cell_2], embedding_layer,  dtype=tf.float32)
    
    x = tf.layers.dense(inputs=x[0][-1], units=params["dense_size"], activation=tf.nn.relu)
    
    output_layer = tf.layers.dense(inputs=x, units=6, activation=tf.nn.sigmoid)
    predictions = output_layer
    
    loss = tf.reduce_mean(tf.losses.log_loss(labels=targets, predictions=output_layer))
    
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=params['learning_rate'],
        optimizer="RMSProp",
        clip_gradients=1.0
    )
    return predictions, loss, train_op
    
