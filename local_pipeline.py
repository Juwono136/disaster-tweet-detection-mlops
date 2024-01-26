"""
Initiate pipeline orchestration
"""

# pylint: disable=[E1123, W0621]
import os
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "juwono136-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/tweet_transform.py"
TRAINER_MODULE_FILE = "modules/tweet_trainer.py"
TUNER_MODULE_FILE = "modules/tweet_tuner.py"

# pipeline outputs
OUTPUT_BASE = "outputs"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, 'metadata.sqlite')


def init_local_pipeline(components, pipeline_root: Text) -> pipeline.Pipeline:
    """Initialize and run a local TFX pipeline using the DirectRunner.

    Args:
        components: Tuple containing TFX components to be included in the pipeline.
        pipeline_root (Text): Root directory for storing pipeline artifacts and metadata.

    Returns:
        pipeline.Pipeline: TFX pipeline configured for local execution.
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path)
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    components = init_components(
        DATA_ROOT,
        transform_module=TRANSFORM_MODULE_FILE,
        training_module=TRAINER_MODULE_FILE,
        tuner_module=TUNER_MODULE_FILE,
        training_steps=500,
        eval_steps=50,
        serving_model_dir=serving_model_dir
    )

    pipeline = init_local_pipeline(components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
