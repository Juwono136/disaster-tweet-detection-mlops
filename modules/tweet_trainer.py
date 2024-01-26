"""
Trainer module
"""

# pylint: disable=E0401
import os
import tensorflow as tf
import tensorflow_transform as tft
import keras_tuner
from keras import layers
from keras.utils.vis_utils import plot_model
from tfx.components.trainer.fn_args_utils import FnArgs

from tweet_transform import (
    FEATURE_KEY,
    LABEL_KEY,
    transformed_name
)


VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

vectorize_layer = layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH
)


def model_builder(hp):
    """Build machine learning model.

    Args:
        hp (keras_tuner.HyperParameters): Hyperparameters for model tuning.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_embedding_dim = hp.Int('embedding_dim', min_value=8, max_value=128)

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])

    x = vectorize_layer(reshaped_narrative)
    x = layers.Embedding(VOCAB_SIZE, hp_embedding_dim, name="embedding")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    # print model
    model.summary()
    return model


def gzip_reader_fn(filenames):
    """Load compressed data.

    Args:
        filenames (list): List of file names.

    Returns:
        tf.data.TFRecordDataset: Dataset containing compressed data.
    """

    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Retrieve the feature transformation layer and define the serving function.

    Args:
        model (tf.keras.Model): Trained Keras model.
        tf_transform_output (tft.TFTransformOutput): Output of the TensorFlow Transform component.

    Returns:
        function: TensorFlow serving function.
    """

    # retrieves the feature transformation layer from the TensorFlow transform
    # output and stores it in the model
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    # defines the TensorFlow serving function to receive serialized input
    # examples and return the model predictions
    def serve_tf_examples_fn(serialized_tf_examples):

        # retrieves the raw feature specifications from the TensorFlow
        # transform output
        feature_spec = tf_transform_output.raw_feature_spec()

        # removes the label key from the features specifications
        feature_spec.pop(LABEL_KEY)

        # parse the serialized examples into a format usable by the model
        # during inference
        parsed_features = tf.io.parse_example(
            serialized_tf_examples, feature_spec)

        # applies the same transformations used during training (tft_layer) to
        # the parsed input examples
        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn


def input_fn(
        file_pattern,
        tf_transform_output,
        num_epochs=None,
        batch_size=64) -> tf.data.Dataset:
    """Get post_transform feature & create batches of data.

    Args:
        file_pattern (str): File pattern to match input files.
        tf_transform_output (tft.TFTransformOutput): Output of the TensorFlow Transform component.
        num_epochs (int, optional): Number of epochs for dataset iteration. Defaults to None.
        batch_size (int, optional): Batch size for the dataset. Defaults to 64.

    Returns:
        tf.data.Dataset: Batched and transformed dataset.
    """

    # get post_transform feature spec
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY))

    return dataset


def run_fn(fn_args: FnArgs) -> None:
    """Define the training and serving function.

    Args:
        fn_args (FnArgs): Arguments passed to the function.
    """

    # defines the directory for TensorBoard logs
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')

    # creates a TensorBoard callback to visualize the training progress
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq='batch'
    )
    # defines an Early Stopping (es) and Model Checkpoint (mc) callback that
    # monitors the validation binary accuracy
    es = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(
        fn_args.serving_model_dir,
        monitor='val_binary_accuracy',
        mode='max',
        verbose=1,
        save_best_only=True)

    # load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # create batches of data
    train_set = input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 10)
    vectorize_layer.adapt(
        [j[0].numpy()[0] for j in [
            i[0][transformed_name(FEATURE_KEY)]
            for i in list(train_set)
        ]]
    )
    # vectorize_layer.adapt(train_set.map(lambda x, y: x[transformed_name(FEATURE_KEY)]))

    # build the model
    hp = keras_tuner.HyperParameters()
    model = model_builder(hp)

    # train the model
    model.fit(x=train_set,
              validation_data=val_set,
              callbacks=[tensorboard_callback, es, mc],
              steps_per_epoch=1000,
              validation_steps=1000,
              epochs=10)

    # defines the serving signatures for the saved model
    signatures = {
        'serving_default': _get_serve_tf_examples_fn(
            model,
            tf_transform_output).get_concrete_function(
            tf.TensorSpec(
                shape=[None],
                dtype=tf.string,
                name='examples'))}

    # save the model
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)

    # save plot model
    plot_model(
        model,
        to_file='images/model_plot.png',
        show_shapes=True,
        show_layer_names=True
    )
