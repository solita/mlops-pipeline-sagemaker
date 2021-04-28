"""A soon-to-be-refactored mess of a pipeline definition"""
from sagemaker.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics

from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep, JsonGet

from config import read_conf

conf = read_conf("../config_bitti.ini")

# data augmentation step
# the data set that was collected is slow, just around fifty images;
# furthermore, all the magazines are placed with the same top-down view
# with labels horizontally aligned. We are going to change that.
magazine_augmentor = ScriptProcessor(
    image_uri=str(conf.processing_turicreate_uri),
    command=["python3"],
    instance_type=conf.processing_instance_type,
    instance_count=conf.processing_instance_count,
    #env={"RotationAngle": "15"},  # TODO: add configurability
    base_job_name="script-magazine-augmentation",
    role=conf.role)

step_data_augmentation = ProcessingStep(
    name="DataAugmentation",
    processor=magazine_augmentor,
    inputs=[
      ProcessingInput(source=conf.input_data,
                      destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="augmented",
                         source="/opt/ml/processing/output")],
    code=f"{conf.source_dir}/augmentation.py",
    cache_config=conf.cache_config)

# preprocessing step
# download the dataset, convert it into Turi Create's SFrame object,
# and save the output on S3 in a train/test split.
sframes_preprocessor = ScriptProcessor(
    image_uri=str(conf.processing_turicreate_uri),
    command=["python3"],
    instance_type=conf.processing_instance_type,
    instance_count=conf.processing_instance_count,
    env={"TrainSplitFraction": str(conf.processing_train_test_split)},
    base_job_name="script-sframe-conversion",
    role=conf.role)

step_sframe_process = ProcessingStep(
    name="BittiDataProcessing",
    processor=sframes_preprocessor,
    inputs=[
      ProcessingInput(
            source=step_data_augmentation.properties.ProcessingOutputConfig.Outputs[
                "augmented"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output_train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/output_test")
    ],
    code=f"{conf.source_dir}/preprocessing.py",
    cache_config=conf.cache_config)

# TODO: this should be ParamString
model_path = f"s3://{conf.bucket}/output_model"

# Regular expressions are a pain, use the playground here:
# https://regex101.com/r/kopij0/1
turicreate_metrics = [
        {'Name': 'train:loss',
         'Regex': r"'train:loss': (?:\| [0-9]+ \| )([0-9]+[.][0-9]+)"}]

tf_train = TensorFlow(base_job_name='bitti-turicreate-pipelines',
                      entry_point='training.py',
                      source_dir=conf.source_dir,
                      output_path=model_path,
                      role=conf.role,
                      image_uri=str(conf.training_turicreate_uri),
                      hyperparameters={
                          'max-iterations': int(conf.training_max_iterations),
                          'batch-size': int(conf.training_batch_size)},
                      instance_count=conf.training_instance_count,
                      instance_type=conf.training_instance_type,
                      metric_definitions=turicreate_metrics,
                      input_mode='File')

# TODO: change into Pipe - but that would need additional read f-ions in training.py
step_train = TrainingStep(
    name="ModelTraining",
    estimator=tf_train,
    inputs={
        "train": TrainingInput(
            step_sframe_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            input_mode="File"),
        "test": TrainingInput(
            step_sframe_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            input_mode="File"),
    },
    cache_config=conf.cache_config)

# define the evaluation step
# we have trained the model, but we need to validate it too; what happened
# the first time was, due to the EXIF tags half of images were not rotated
# properly relative to the labels... needless to say, that led to the mAP
# score to be close to nill; that's a good example for why the validation
# step is needed - only models that work should be carried on with.
script_eval = ScriptProcessor(
    image_uri=str(conf.processing_turicreate_uri),
    command=["python3"],
    instance_type=conf.processing_instance_type,
    instance_count=conf.processing_instance_count,
    base_job_name="script-bitti-eval",
    role=conf.role)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json")

step_eval = ProcessingStep(
    name="ModelEvaluation",
    processor=script_eval,
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"),
        ProcessingInput(
            source=step_sframe_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/test")],
    outputs=[
        ProcessingOutput(output_name="evaluation",
                         source="/opt/ml/processing/evaluation"),
        ProcessingOutput(output_name="mlmodel",
                         source="/opt/ml/processing/mlmodel"),
        ProcessingOutput(output_name="eval_images",
                         source="/opt/ml/processing/eval_images")],
    code=f"{conf.source_dir}/evaluation.py",
    property_files=[evaluation_report])

cond_map = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step=step_eval,
        property_file=evaluation_report,
        json_path="regression_metrics.mAP.value"),
    right=conf.model_approval_map_threshold)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri="{}/evaluation.json".format(
            step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        ),
        content_type="application/json"))

step_register = RegisterModel(
    name="BittiRegisterModel",
    estimator=tf_train,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/octet-stream"],
    response_types=["application/octet-stream"],
    inference_instances=["ml.t2.medium", "ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name=conf.model_package_group_name,
    approval_status=conf.model_approval_status,
    model_metrics=model_metrics)

script_publish = ScriptProcessor(
    image_uri=str(conf.processing_turicreate_uri),
    command=["python3"],
    instance_type=conf.processing_instance_type,
    instance_count=1,
    base_job_name="script-bitti-publish",
    role=conf.role)

# model card generator
# absolute wild west of a pipleine step
model_card_generator = ScriptProcessor(
    image_uri=str(conf.summarizing_turicreate_uri),
    command=["python3"],
    instance_type=conf.summarizing_instance_type,
    instance_count=conf.summarizing_instance_count,
    base_job_name="script-model-card",
    role=conf.role)

step_model_card = ProcessingStep(
    name="ModelCardGenerator",
    processor=model_card_generator,
    inputs=[
        ProcessingInput(
            source=step_eval.properties.ProcessingOutputConfig.Outputs[
                "eval_images"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/eval_images"),
        ProcessingInput(
            source=step_eval.properties.ProcessingOutputConfig.Outputs[
                "evaluation"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/evaluation")],
    outputs=[
        ProcessingOutput(output_name="model_card",
                         source="/opt/ml/processing/model_card")],
    code=f"{conf.source_dir}/model_card.py",
    cache_config=conf.cache_config)

step_publish = ProcessingStep(
    name="PublishViaAPI",
    processor=script_publish,
    inputs=[
        ProcessingInput(
            source=step_eval.properties.ProcessingOutputConfig.Outputs[
                "mlmodel"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/mlmodel"),
        ProcessingInput(
            source=step_model_card.properties.ProcessingOutputConfig.Outputs[
                "model_card"
            ].S3Output.S3Uri,
            destination="/opt/ml/processing/model_card")
        ],
    code=f"{conf.source_dir}/publish_to_api.py")

step_cond = ConditionStep(
    name="BittymAPcheck",
    conditions=[cond_map],
    if_steps=[step_register, step_publish],
    else_steps=[])

# finally, putting all the steps together in a Pipeline instance
pipeline_name = conf.pipeline_name
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[
        conf.processing_train_test_split,
        conf.processing_instance_count,
        conf.processing_instance_type,
        conf.summarizing_instance_count,
        conf.summarizing_instance_type,
        conf.training_instance_count,
        conf.training_instance_type,
        conf.training_batch_size,
        conf.training_max_iterations,
        conf.model_approval_status,
        conf.input_data,
        conf.model_approval_map_threshold
    ],
    steps=[step_data_augmentation, step_sframe_process,
           step_train, step_eval, step_cond, step_model_card])
