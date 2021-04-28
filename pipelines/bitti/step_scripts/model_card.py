"""Proof-of-concept for model card integration in SageMaker Pipelines"""
import logging
from pathlib import Path
from datetime import datetime
import dataclasses
from typing import List, Text, Union
import json
import base64
import boto3
import model_card_toolkit

logging.getLogger().setLevel(logging.INFO)

BASE_DIR = Path("/opt/ml/processing").resolve()
EVAL_REPORT_DIR = BASE_DIR/"evaluation"
EVAL_IMAGES_DIR = BASE_DIR/"eval_images"
OUTPUT_DIR = BASE_DIR/"model_card"

# TODO: if putting this into its own step, we'll need to find the
# current execution ARN (pipelines have concurrency so checking on status
# won't cut it). The only way I can see so far is to inject a fingerprint
# into, e.g. evaluation report and then match on fingerprints of all running
# pipelines. This should introduce a couple of extra S3 calls but that's
# alright as time is negligible in the overall pipeline exec time.
# ... for now, take the oldest running pipeline


@dataclasses.dataclass
class OperationalSetting:
    """Operational setting of the pipeline.

    Attributes:
      type: A short name / description of the pipeline setting
      value: Value used in the pipeline execution
    """
    type: Text
    value: Union[int, float, Text]


@dataclasses.dataclass
class PipelineParameters:
    """Parameters of pipeline's execution run.

    Attributes:
      pipeline_parameters: List of pipeline parameter values.
    """
    pipeline_parameters: List[OperationalSetting] = dataclasses.field(
        default_factory=list)


def _fetch_def_parval(parsed_json, parname):
    matches = [p['DefaultValue'] for p in parsed_json['Parameters']
               if p['Name'] == parname]
    return matches[0] if matches else None


pipeline_name = "BittiPipeline"
sensitive_data = False

region_name = "eu-central-1"
session = boto3.session.Session()
client_sagemaker = session.client(
    service_name='sagemaker',
    region_name=region_name
)
pipe_executions = client_sagemaker.list_pipeline_executions(
        PipelineName=pipeline_name)

current_date = datetime.now(tz=datetime.now().astimezone().tzinfo)
last_exec, last_start_time, last_exec_arn = None, current_date, None
for exec_dict in pipe_executions['PipelineExecutionSummaries']:
    if (exec_dict['PipelineExecutionStatus'] == 'Executing'
            and exec_dict['StartTime'] < last_start_time):
        last_exec_arn = exec_dict['PipelineExecutionArn']
        last_start_time = exec_dict['StartTime']
        last_exec = exec_dict['PipelineExecutionDisplayName']
        last_description = exec_dict['PipelineExecutionDescription']

if not last_exec:
    raise RuntimeError("Can't figure out which pipeline execution this"
                       " pipeline step is running in!")

logging.info("Selected pipeline run: %s", last_exec)
logging.info("Selected pipeline arn: %s", last_exec_arn)
logging.info("Selected pipeline ran: %s", last_start_time)

pipe_details = client_sagemaker.describe_pipeline_definition_for_execution(
        PipelineExecutionArn=last_exec_arn)
pipe_definition = json.loads(pipe_details["PipelineDefinition"])
step_dict = {p['Name']: p for p in pipe_definition['Steps']}

logging.info("Pipeline definition: %s", pipe_details["PipelineDefinition"])

train_test_split = _fetch_def_parval(pipe_definition, 'TrainTestSplit')
s3_input_data = _fetch_def_parval(pipe_definition, 'InputData')
training_batch_size = _fetch_def_parval(pipe_definition, 'TrainingBatchSize')
training_instance_type = _fetch_def_parval(pipe_definition,
                                           'TrainingInstanceType')
max_iterations = _fetch_def_parval(pipe_definition, 'MaxIterations')
model_approval_map_cut = _fetch_def_parval(pipe_definition,
                                           'ModelApprovalmAPThreshold')
with open(EVAL_REPORT_DIR/"evaluation.json", "r") as fin:
    eval_data = json.load(fin)

mAP = model_card_toolkit.PerformanceMetric(
        type='Mean average precision (mAP) score',
        value=f"{eval_data['regression_metrics']['mAP']['value']*100:.2f}%")

# TODO: some of the values below can be obtained from .mlmodel file
#       metadata. But for that I'll need to include coremltools alongside
#       the model-card-toolkit - still have to check for compatible versions.
# TODO: the rest of the - very voluminous! - descriptions should be migrated
#       into the .ini file and used as pipeline parameters perhaps?
mct = model_card_toolkit.ModelCardToolkit()
model_card = mct.scaffold_assets()
model_card.model_details.name = pipeline_name
model_card.model_details.overview = (
        "This is an explainability report supplementing a magazine logo"
        " detector neural network model. This model card is generated"
        " automatically as part of the AWS SageMaker Pipelines execution run"
        f" (version '{last_exec}') that trained the accompanying model"
        " version. The pipeline took the input training images and"
        " annotations (available for Solita's internal use at"
        f" {s3_input_data}), augmented each image/label pair with a hundred"
        " random rotation and projection transformation, and fed the resulting"
        " data, split into training and evaluation set, into the training"
        " script. The training, executed as a SageMaker Training Job with a"
        f" custom Turi Create ECR image, achieved a mAP score of {mAP.value} on"
        " BITTI magazine labels. After cross-checking with the mAP cutoff of"
        f" {model_approval_map_cut*100:.2f}%, the model card generator step of"
        " the pipeline created this HTML file and deployed it as a"
        " publicly available static webpage.")
model_card.model_details.owners = [
        model_card_toolkit.Owner("Solita Oy", "AIGA WP3 working group"),
        model_card_toolkit.Owner("Vlas Sokolov", "vlas.sokolov@solita.fi")]
model_card.model_details.version.name = last_exec
model_card.model_details.version.date = str(last_start_time)
model_card.model_details.version.diff = last_description
model_card.model_details.license = "MIT License"

s3_train_data = step_dict['BittiDataProcessing']['Arguments'][
        'ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
mct_data = model_card.model_parameters.data
mct_data.train.link = s3_train_data
mct_data.train.sensitive = sensitive_data
mct_data.train.name = (
        f"Bitti training data ({train_test_split*100:.1f}% split)")
s3_test_data = step_dict['BittiDataProcessing']['Arguments'][
        'ProcessingOutputConfig']['Outputs'][1]['S3Output']['S3Uri']
mct_data.eval.link = s3_test_data
mct_data.eval.sensitive = sensitive_data
mct_data.eval.name = (
        f"Bitti evaluation data ({(1-train_test_split)*100:.1f}% split)")
# add evaluation example graphics
mct_data.eval.graphics.description = (
        "The images below show the model performance on the evaluation set."
        " The randomly chosen images and overlaid with model predictions.")
mct_data.eval.graphics.collection = []
for i, img_path in enumerate(EVAL_IMAGES_DIR.glob("*")):
    with open(img_path, "rb") as image_file:
        img_str_base64 = base64.b64encode(image_file.read())
    graphic = model_card_toolkit.Graphic(name=f"Evaluation image #{i+1}",
                                         image=img_str_base64.decode())
    mct_data.eval.graphics.collection.append(graphic)

map_threshold = _fetch_def_parval(pipe_definition, 'ModelApprovalmAPThreshold')
mct_data.input = model_card_toolkit.Dataset(
        name="input", link=s3_input_data, sensitive=sensitive_data)
mct_data.augmented = model_card_toolkit.Dataset(
        name="augmented", sensitive=sensitive_data,
        link=step_dict['DataAugmentation']['Arguments'][
            'ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri'])

turicreate_object_detection_uri = (
        "https://github.com/apple/turicreate/blob/"
        "master/userguide/object_detection/how-it-works.md")
# TODO: figure out how to pass Jinja's {{ value | safe }} here
# model_card.model_parameters.model_architecture = (
#         "TinyYOLO re-implemneted in turicreate. Extensive writeup"
#         " about the turicreate implementation can be found"
#         f" here: <a href=\"{turicreate_object_detection_uri}\">"
#         "How TuriCreate object detection works.</a>")
model_card.model_parameters.model_architecture = (
        "TinyYOLO re-implemneted in turicreate. Extensive writeup"
        " about the turicreate implementation can be found"
        f" here: {turicreate_object_detection_uri}")

model_card.quantitative_analysis.performance_metrics.append(mAP)
model_card.quantitative_analysis.performance_metrics.extend([
        model_card_toolkit.PerformanceMetric(
          type='Fraction of data reserved for training',
          value=train_test_split),
        model_card_toolkit.PerformanceMetric(
          type='Training: maximum number of iterations',
          value=max_iterations),
        model_card_toolkit.PerformanceMetric(
          type='Training: batch size',
          value=training_batch_size),
        model_card_toolkit.PerformanceMetric(
          type='Training: instance type',
          value=training_instance_type),
        model_card_toolkit.PerformanceMetric(
          type='Model approval mAP threshold',
          value=model_approval_map_cut),
])

model_card.considerations.users.append(
    "Potential MLOps users looking for a CI/CD template for model deployment."
)
model_card.considerations.users.append(
    "iOS developers interested in integrating pipeline-resultant"
    " models into iPhone applications."
)
model_card.considerations.users.append(
    "iPhone application users who seek to get insights into how the"
    " neural network model was created."
)
model_card.considerations.users.append(
    "AI explainability enthusiasts and researchers interested in"
    " an xAI reference pipeline."
)
model_card.considerations.use_cases.append(
    "The provided model is intended to be used as a part of a BITTI"
    " magazine logo recognition application."
)
model_card.considerations.use_cases.append(
    "It was developed for demo purposes, and was not intended to achieve"
    " performance comparable to with state of the art computer vision"
    " and deep learning applications."
)
model_card.considerations.limitations.append(
    "The model was trained to recognize logos that roughly align with the"
    " camera (in about +/- 15 degrees range), tilting the logos beyond"
    " that range will likely result in degraded performance."
)
model_card.considerations.limitations.append(
    "The model was trained to recognize logos with minimal projection"
    " distortion. Using it on an image that was, for example, taken from"
    " the side, will often not give desirable results."
)
model_card.considerations.limitations.append(
    "Far away, blurry, obscured, over- or under-exposed magazine logo"
    " images affect model performance negatively."
)
model_card.considerations.tradeoffs.append(
    "The architecture chosen for the model was designed to process"
    " smartphone camera videos in real-time. As only a limited processing"
    " power is expected to be available, the trained model is relatively"
    " lightweight. This, in turn, means that the model is not capable of"
    " competing with state-of-the-art object detection architectures"
    " of larger size."
)
model_card.considerations.tradeoffs.append(
    "While other magazine labels can be learned from the same training set,"
    " the developers chose not to label them in the training dataset due"
    " to time constraints."
)
model_card.considerations.ethical_considerations = [
    model_card_toolkit.Risk(
        name=(
            "As the magazine images are in public domain, there are few"
            " ethical considerations the model developers can think of."
            " However, potential model users or applications that take, e.g.,"
            " magazine rack images in a store should be wary of accidentally"
            " recording or leaking sensitive data such as people imagery or"
            " store pricing/stock information."),
        mitigation_strategy=(
            "As the model is fully capable to run on an edge device such as a"
            " mobile phone, a mitigation strategy for the issue above is to"
            " limit the data storage to that of a local device."))]

mct.update_model_card_json(model_card)
html = mct.export_format()

logging.info("Created a model card HTML: \n%s", html)
with open(OUTPUT_DIR/"model_card.html", "w") as fout:
    fout.write(html)
