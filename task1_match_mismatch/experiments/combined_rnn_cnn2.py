import tensorflow as tf
from tensorflow.keras.layers import Dropout, Input, Conv1D, LSTM, Dense, Flatten, Concatenate

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

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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

def combined_model2(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    cnn_layers=2,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    lstm_units=32,
    num_mismatched_segments=2,
    activation="relu",
    dropout_rate=0.3,  # Add dropout rate for regularization
    compile=True
):
    # EEG branch
    eeg_input = Input(shape=[time_window, eeg_input_dimension])
    eeg_conv = Conv1D(32, kernel_size=3, activation='relu')(eeg_input)
    eeg_lstm = LSTM(64)(eeg_conv)
    eeg_output = Dense(64, activation='relu')(eeg_lstm)
    eeg_output = Dropout(dropout_rate)(eeg_output)  # Adding dropout for regularization

    # Stimuli branches
    stimuli_inputs = [Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments + 1)]
    stimuli_projections = [Conv1D(32, kernel_size=3, activation='relu')(stimuli) for stimuli in stimuli_inputs]
    stimuli_lstm = [LSTM(64)(stimuli_proj) for stimuli_proj in stimuli_projections]

    # Concatenate EEG and Stimuli branches
    combined = Concatenate()([eeg_output] + stimuli_lstm)
    combined = Dropout(dropout_rate)(combined)  # Adding dropout for regularization

    # Fully connected layers
    dense_1 = Dense(128, activation='relu')(combined)
    dense_1 = Dropout(dropout_rate)(dense_1)  # Adding dropout for regularization
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_2 = Dropout(dropout_rate)(dense_2)  # Adding dropout for regularization

    # Output layer
    output = Dense(5, activation='softmax')(dense_2)

    # Create and compile the model
    model = tf.keras.Model(inputs=[eeg_input] + stimuli_inputs, outputs=output)
            
    if compile:
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
            run_eagerly=True,
        )
        print(model.summary())

    return model


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length_s = 5
    fs = 64

    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64

    epochs = 100
    patience = 5
    batch_size = 64
    only_evaluate = False
    number_mismatch = 4 # or 4



    training_log_filename = "training_log_{}_{}.csv".format(number_mismatch, window_length_s)



    # Get the path to the config gile
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    util_folder = os.path.join(os.path.dirname(task_folder), "util")
    #config_path = os.path.join(util_folder, 'config.json')
    #config_path = "auditory-eeg-challenge-2024-code/util/config.json"
    # Load the config
    #with open(config_path) as fp:
        #config = json.load(fp)

    data_folder = "/scratch/at5282/split_data/"

    # Provide the path of the dataset
    # which is split already to train, val, test
    # data_folder = os.path.join(config["dataset_folder"], config['derivatives_folder'], config["split_folder"])

    #data_folder = "/content/split_data"

    # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 28)
    stimulus_features = ["envelope"]
    stimulus_dimension = 1

    # uncomment if you want to train with the mel spectrogram stimulus representation
    # stimulus_features = ["mel"]
    # stimulus_dimension = 10

    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder, "combined_model_{}_MM_{}_s_{}".format(number_mismatch, window_length_s, stimulus_features[0]))
    os.makedirs(results_folder, exist_ok=True)

    # create dilation model
    model = combined_model2(time_window=window_length, eeg_input_dimension=64, env_input_dimension=1, num_mismatched_segments = number_mismatch)

    model_path = os.path.join(results_folder, "model_{}_MM_{}_s_{}.h5".format(number_mismatch, window_length_s, stimulus_features[0]))

    if only_evaluate:
        model = tf.keras.models.load_model(model_path)

    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        import pdb
        dataset_train = create_tf_dataset(train_generator, window_length, batch_equalizer_fn,
                                          hop_length, batch_size,
                                          number_mismatch=number_mismatch,
                                          data_types=(tf.float32, tf.float32),
                                          feature_dims=(64, stimulus_dimension))

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

            model = combined_model2(time_window=window_length, eeg_input_dimension=64,
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
