# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:48:48 2023

@author: SimonKerner
"""

import tensorflow as tf
import numpy as np
import regex as re
import subprocess
import os

from IPython import display as ipythondisplay
from matplotlib import pyplot as plt
from IPython.display import Audio
from tqdm import tqdm


################################################################################################
# Dataset functions
################################################################################################


CWD = os.getcwd()


def load_data(dataset):
    
    path = os.path.join(CWD, 'data', '{}.abc'.format(dataset))
    
    with open(path, 'r') as f:
        file = f.read()
        
    songs = extract_song_snippet(file)
    
    print('There are {} songs in the "{}" dataset.'.format(len(songs), dataset))
    
    return songs




def extract_song_snippet(text):
    
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    
    return songs




def get_vocabulary(abc_songs):

    global vocabulary
    vocabulary = dict(enumerate(sorted(set("".join(abc_songs)))))
    vocabulary = {v: k for k, v in vocabulary.items()}
    
    return vocabulary




def string2numerical(song_string):
    
    numerical = [vocabulary.get(i) for i in song_string]
    
    return np.array(numerical)




def numerical2string(numerical_song):
    
    inverse_vocab = {v: k for k, v in vocabulary.items()}
    
    return "".join([inverse_vocab.get(i) for i in numerical_song])




def create_batch(vectorized_songs, seq_length, batch_size):
    
    #get random start-index of vectorized_songs
    index = np.random.choice(len(vectorized_songs) - 1 - seq_length, batch_size)
    
    # get batch for sequenze length out of vectorized_songs
    input_batch = [vectorized_songs[i : i + seq_length] for i in index]
    output_batch = [vectorized_songs[i + 1 : i + seq_length + 1] for i in index]
    
    # x_batch --> y_batch
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    
    return x_batch, y_batch




def generate_text(model, start_string, generation_length):
    
    input_eval = string2numerical(start_string)
    input_eval = tf.expand_dims(input_eval, 0)
    
    generated_text = []
    
    model.reset_states()
    tqdm._instances.clear()
    
    for i in range(generation_length):
        
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        
        prdicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        input_eval = tf.expand_dims([prdicted_id], 0)
        
        generated_text.append(numerical2string([prdicted_id]))
    
    
    return (start_string + ''.join(generated_text))




def generate_songs(model, songcount, generation_length=1000):
    
    # start seed for input evaluation
    START_STRING = 'X'
    generated_songs = []
    
    for i in tqdm(range(songcount)):
        generated_text = generate_text(model, START_STRING, generation_length)
        extracted_songs = extract_song_snippet(generated_text)

        if extracted_songs:
            generated_songs.append(max(extracted_songs))
        
    return generated_songs




################################################################################################
# Audio functions for playing music
################################################################################################




def save_song_to_abc(song):

    # save file to abc
    path = os.path.join(CWD, 'songs', 'temp.abc')

    with open(path, "w") as f:
        f.write(song)
        


def show_songs():
    
    path = os.path.join(CWD, 'songs')
    
    for file_name in os.listdir(path):
        
        if file_name.endswith(".wav"):
            print(file_name)

        
        
                
def delete_songs():
    
    path = os.path.join(CWD, 'songs')
    
    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))
    
    print("Files deleted successfully!")
    



def play_song(songnumber):
    
    return Audio(filename='songs/song_' + str(songnumber) + '.wav')




def create_wav_songs(song, songnumber):
    
    # crete temporary file and save to it abc
    save_song_to_abc(song)
    
    # create necessary song files --> .abc to .wav
    subprocess.run('wsl abc2midi temp.abc -o temp.mid', cwd=os.path.join(CWD, 'songs'))
    subprocess.run('wsl timidity {}.mid -Ow -o {}.wav'.format('temp', 'song_' + str(songnumber)), 
                   cwd=os.path.join(CWD, 'songs'))
    



def play_generated_songs(generated_songs, print_abc=True):
    
    if len(generated_songs) == 0:
        return print('No valid songs found!')

    for i, song in enumerate(generated_songs):
        
        print('Generated Song:', i)
        create_wav_songs(song, i)
        
        if print_abc:
            print("\n", song)
        
        ipythondisplay.display(play_song(i))
        print()




################################################################################################
# The Recurrent Neural Network
################################################################################################




def LSTM(rnn_units):
    
    return tf.keras.layers.LSTM(rnn_units,
                                return_sequences=True,
                                recurrent_initializer='glorot_uniform',
                                recurrent_activation='sigmoid',
                                stateful=True
                                )




def build_model(vocabulary_size, embedding_dim, rnn_units, batch_size):
    
    model = tf.keras.Sequential([
        
        # Layer 1: Embedding layer to transform indices into dense vectors 
        tf.keras.layers.Embedding(vocabulary_size, embedding_dim, batch_input_shape=[batch_size, None]),
        
        # Layer 2: LSTM with `rnn_units` number of units.
        LSTM(rnn_units),
        
        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
        #          into the vocabulary size.
        tf.keras.layers.Dense(vocabulary_size)
        ])
    
    return model




def compute_loss(labels, logits):
    
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
    return loss




@tf.function
def train_step(x, y, model, optimizer):
    
    with tf.GradientTape() as tape:
        
        y_hat = model(x)
        loss = compute_loss(y, y_hat)
        
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss




def plot_loss(history):
    
    plt.cla()
    plt.plot(history)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    
    ipythondisplay.clear_output(wait=True)
    ipythondisplay.display(plt.gcf())
    
    plt.close()
        
    


def train_model(num_iterations, vectorized_songs, seq_length, batch_size, model, optimizer, checkpoint_prefix, plot_loss=True):
    
    history = []

    for iter in tqdm(range(num_iterations)):
        
        x_batch, y_batch = create_batch(vectorized_songs, seq_length, batch_size)
        loss = train_step(x_batch, y_batch, model, optimizer)
        
        history.append(loss.numpy().mean())
        
        if plot_loss == True:
            
            plot_loss(history)
        
        # create checkpoints during training
        if iter % 100 == 0:
            model.save_weights(checkpoint_prefix)
    
    model.save_weights(checkpoint_prefix)



