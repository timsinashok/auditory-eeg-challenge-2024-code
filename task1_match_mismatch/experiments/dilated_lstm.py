import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Flatten, Concatenate, BatchNormalization, Dropout, GlobalAveragePooling1D

print("*********improved_dilation_model_with_lstm****************")

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, LSTM, BatchNormalization, Dropout, Dense, Flatten, Input, Dot, Concatenate
"""Example experiment for the 2 mismatched segments dilation model."""
import glob
import json
import logging
import os, sys
import tensorflow as tf

import sys
# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from util.dataset_generator import DataGenerator, batch_equalizer_fn, create_tf_dataset

#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, BatchNormalization, Concatenate

### new model did not perfomr as good
## last one trained in AD
def improved_dilation_model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    lstm_units=128,
    num_mismatched_segments=2,
    dropout_rate=0.3,  # Adjusted dropout rate
    compile=True
):
    eeg = Input(shape=[time_window, eeg_input_dimension])
    stimuli_input = [Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments + 1)]

    all_inputs = [eeg]
    all_inputs.extend(stimuli_input)

    stimuli_proj = [x for x in stimuli_input]

    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    # Spatial convolution
    eeg_proj_1 = Conv1D(spatial_filters, kernel_size=1)(eeg)
    eeg_proj_1 = BatchNormalization()(eeg_proj_1)

    # LSTM layer
    eeg_proj_1 = LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)(eeg_proj_1)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(eeg_proj_1)
        eeg_proj_1 = BatchNormalization()(eeg_proj_1)
        eeg_proj_1 = Dropout(dropout_rate)(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        stimuli_proj = [env_proj_layer(stimulus_proj) for stimulus_proj in stimuli_proj]

    # Comparison
    cos = [Dot(1, normalize=True)([eeg_proj_1, stimulus_proj]) for stimulus_proj in stimuli_proj]

    linear_proj_sim = Dense(1, activation="linear")

    # Linear projection of similarity matrices
    cos_proj = [linear_proj_sim(Flatten()(cos_i)) for cos_i in cos]

    # Classification
    out = tf.keras.activations.softmax((Concatenate()(cos_proj)))

    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
        print(model.summary())
    return model


## LSTM added dilated without normalization performed little less better than normalized one
def improved_dilation_model_with_lstm(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    lstm_units=256,  # Increase LSTM units
    dense_units=256,  # Increase dense layer size
    dropout_rate=0.5,  # Adjusted dropout rate
    num_mismatched_segments=2,
    compile=True
):
    eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
    stimuli_input = [tf.keras.layers.Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments + 1)]

    all_inputs = [eeg]
    all_inputs.extend(stimuli_input)

    stimuli_proj = [x for x in stimuli_input]

    # Spatial convolution
    eeg_proj_1 = tf.keras.layers.Conv1D(spatial_filters, kernel_size=1)(eeg)

    # LSTM layer
    eeg_proj_1 = LSTM(lstm_units, return_sequences=True)(eeg_proj_1)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activation,
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activation,
        )

        stimuli_proj = [env_proj_layer(stimulus_proj) for stimulus_proj in stimuli_proj]

    # Comparison
    cos = [tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, stimulus_proj]) for stimulus_proj in stimuli_proj]

    # Linear projection of similarity matrices
    cos_proj = [tf.keras.layers.Dense(1, activation="linear")(tf.keras.layers.Flatten()(cos_i)) for cos_i in cos]

    # Concatenate the linear projections
    concat_cos = tf.keras.layers.Concatenate()(cos_proj)

    # Fully connected layers
    dense_1 = Dense(dense_units, activation='relu')(concat_cos)
    dense_1 = BatchNormalization()(dense_1)
    dense_1 = Dropout(dropout_rate)(dense_1)

    # Output layer
    out = Dense(5, activation='softmax')(dense_1)

    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
        print(model.summary())
    
    return model

## LSTM adddded dilated, 62% accuracy
def dilation_model_with_lstm_updated(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    lstm_units=128,
    num_mismatched_segments=2,
    compile=True
):
    eeg = Input(shape=[time_window, eeg_input_dimension])
    stimuli_input = [Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments + 1)]

    all_inputs = [eeg]
    all_inputs.extend(stimuli_input)

    stimuli_proj = [x for x in stimuli_input]

    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    # Spatial convolution
    eeg_proj_1 = Conv1D(spatial_filters, kernel_size=1)(eeg)
    eeg_proj_1 = BatchNormalization()(eeg_proj_1)

    # LSTM layer
    eeg_proj_1 = LSTM(lstm_units, return_sequences=True)(eeg_proj_1)
    eeg_proj_1 = Dropout(0.2)(eeg_proj_1)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )(eeg_proj_1)
        eeg_proj_1 = BatchNormalization()(eeg_proj_1)
        eeg_proj_1 = Dropout(0.2)(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        stimuli_proj = [env_proj_layer(stimulus_proj) for stimulus_proj in stimuli_proj]

    # Comparison
    cos = [Dot(1, normalize=True)([eeg_proj_1, stimulus_proj]) for stimulus_proj in stimuli_proj]

    linear_proj_sim = Dense(1, activation="linear")

    # Linear projection of similarity matrices
    cos_proj = [linear_proj_sim(Flatten()(cos_i)) for cos_i in cos]

    # Classification
    out = tf.keras.activations.softmax((Concatenate()(cos_proj)))

    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
        print(model.summary())
    return model

def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length_s = 5
    fs = 64

    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64

    epochs = 200
    patience = 5
    batch_size = 64
    only_evaluate = False
    number_mismatch = 4 # or 4



    training_log_filename = "training_log_{}_{}.csv".format(number_mismatch, window_length_s)



    # Get the path to the config gile
     # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    #config_path = os.path.join(util_folder, 'config.json')
    #config_path = "auditory-eeg-challenge-2024-code/util/config.json"
    # Load the config
    #with open(config_path) as fp:
        #config = json.load(fp)

    data_folder = "/scratch/at5282/test_data"
    # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 28)
    stimulus_features = ["envelope"]
    stimulus_dimension = 1

    # uncomment if you want to train with the mel spectrogram stimulus representation
    # stimulus_features = ["mel"]
    # stimulus_dimension = 10

    features = ["eeg","eeg_temporal_jittering","envelope_temporal_jittering", "eeg_time_warpping", "envelope_time_warping"] + stimulus_features
    # Create a directory to store (intermediate) results

    features = ["eeg"] + stimulus_features
    results_folder = os.path.join(experiments_folder, "combined_model_{}_MM_{}_s_{}".format(number_mismatch, window_length_s, stimulus_features[0]))
    os.makedirs(results_folder, exist_ok=True)

    # create dilation model
    model = dilation_model_with_lstm_updated(time_window=window_length, eeg_input_dimension=64, env_input_dimension=1, num_mismatched_segments = number_mismatch)
    tf.config.run_functions_eagerly(True)
    model_path = os.path.join(results_folder, "model_{}_MM_{}_s_{}.h5".format(number_mismatch, window_length_s, stimulus_features[0]))

    if only_evaluate:
        model = tf.keras.models.load_model(model_path)

    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features[:1]]
        print(len(train_files))
        print(train_files[0])
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        import pdb
        dataset_train = create_tf_dataset(train_generator, window_length, batch_equalizer_fn,
                                          hop_length, batch_size,
                                          number_mismatch=number_mismatch,
                                          data_types=(tf.float32, tf.float32),
                                          feature_dims=(64, stimulus_dimension))
        print("no error here")
        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator,  window_length, batch_equalizer_fn,
                                          hop_length, batch_size,
                                          number_mismatch=number_mismatch,
                                          data_types=(tf.float32, tf.float32),
                                          feature_dims=(64, stimulus_dimension))


        # Train the model
        model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ],
        )

    test_window_lengths = [3,5]
    number_mismatch_test = [2,3,4, 8]
    for number_mismatch in number_mismatch_test:
        for window_length_s in test_window_lengths:
            window_length = window_length_s * fs
            results_filename = 'eval_{}_{}_s.json'.format(number_mismatch, window_length_s)

            model = dilation_model_with_lstm_updated(time_window=window_length, eeg_input_dimension=64,
                                   env_input_dimension=stimulus_dimension, num_mismatched_segments=number_mismatch)

            model.load_weights(model_path)
            # Evaluate the model on test set
            # Create a dataset generator for each test subject
            test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if
                          os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
            # Get all different subjects from the test set
            subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
            datasets_test = {}
            # Create a generator for each subject
            for sub in subjects:
                files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
                test_generator = DataGenerator(files_test_sub, window_length)
                datasets_test[sub] = create_tf_dataset(test_generator, window_length, batch_equalizer_fn,
                                                       hop_length, batch_size=1,
                                                       number_mismatch=number_mismatch,
                                                       data_types=(tf.float32, tf.float32),
                                                       feature_dims=(64, stimulus_dimension))

            evaluation = evaluate_model(model, datasets_test)

            # We can save our results in a json encoded file
            results_path = os.path.join(results_folder, results_filename)
            with open(results_path, "w") as fp:
                json.dump(evaluation, fp)
            logging.info(f"Results saved at {results_path}")


