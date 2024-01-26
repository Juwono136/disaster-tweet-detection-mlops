"""
Transform module
"""

import tensorflow as tf

FEATURE_KEY = 'text'
LABEL_KEY = 'target'


def transformed_name(key):
    """Rename transformed features.

    Args:
        key (str): The original feature key.

    Returns:
        str: The transformed feature key.
    """

    return key + "_xf"


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features

    Args:
        inputs (dict): A dictionary containing input features.

    Returns:
        dict: A dictionary containing preprocessed features to transformed features.
    """

    outputs = {}

    # lowercase transformation using TensorFlow
    text = tf.strings.lower(inputs[FEATURE_KEY])

    # removing url using TensorFlow
    text = tf.strings.regex_replace(
        text, r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '')

    # removing non-alphabetic characters using TensorFlow
    text = tf.strings.regex_replace(text, r'[^a-z]', ' ')

    # stripping leading and trailing spaces
    text = tf.strings.strip(text)

    outputs[transformed_name(FEATURE_KEY)] = text

    # Casting label to integer
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
