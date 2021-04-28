"""Proof-of-concept for model card integration in SageMaker Pipelines"""
import logging
from pathlib import Path
from datetime import datetime
import dataclasses
from typing import List, Text, Union
import json
import boto3
import model_card_toolkit

logging.getLogger().setLevel(logging.INFO)

BASE_DIR = Path("/opt/ml/processing").resolve()
EVAL_REPORT_DIR = BASE_DIR/"evaluation"
OUTPUT_DIR = BASE_DIR/"model_card"

# TODO: if putting this into its own step, we'll need to find the
# current execution ARN (pipelines have concurrency so checking on status
# won't cut it). The only way I can see so far is to inject a fingerprint
# into, e.g. evaualtion report and then match on fingerprints of all running
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
      pipeline_parameters: List of pipeline parapeter values.
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

mct = model_card_toolkit.ModelCardToolkit()
model_card = mct.scaffold_assets()
model_card.model_details.name = f"{pipeline_name}Model"
model_card.model_details.overview = "Lorem ipsum"
model_card.model_details.version.name = last_exec
model_card.model_details.version.name = last_exec
model_card.model_details.version.date = str(last_start_time)

train_test_split = _fetch_def_parval(pipe_definition, 'TrainTestSplit')
s3_input_data = _fetch_def_parval(pipe_definition, 'InputData')
s3_train_data = step_dict['BittiDataProcessing']['Arguments'][
        'ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri']
model_card.model_parameters.data.train.link = s3_train_data
model_card.model_parameters.data.train.sensitive = sensitive_data
model_card.model_parameters.data.train.name = (
        f"Bitti training data ({train_test_split*100:.1f}% split)")
s3_test_data = step_dict['BittiDataProcessing']['Arguments'][
        'ProcessingOutputConfig']['Outputs'][1]['S3Output']['S3Uri']
model_card.model_parameters.data.eval.link = s3_test_data
model_card.model_parameters.data.eval.sensitive = sensitive_data
model_card.model_parameters.data.eval.name = (
        f"Bitti evaluation data ({(1-train_test_split)*100:.1f}% split)")
map_threshold = _fetch_def_parval(pipe_definition, 'ModelApprovalmAPThreshold')
model_card.model_parameters.data.input = model_card_toolkit.Dataset(
        name="input", link=s3_input_data, sensitive=sensitive_data)
model_card.model_parameters.data.augmented = model_card_toolkit.Dataset(
        name="augmented", sensitive=sensitive_data,
        link=step_dict['DataAugmentation']['Arguments'][
            'ProcessingOutputConfig']['Outputs'][0]['S3Output']['S3Uri'])

turicreate_object_detection_uri = (
        "https://github.com/apple/turicreate/blob/"
        "master/userguide/object_detection/how-it-works.md")
model_card.model_parameters.model_architecture = (
        "TinyYOLO re-implemneted in turicreate. Extensive writeup"
        " about the turicreate implementation can be found"
        f" here: {turicreate_object_detection_uri}")

with open(EVAL_REPORT_DIR/"evaluation.json", "r") as fin:
    eval_data = json.load(fin)

#eval_report = step_dict['BittymAPcheck']['Arguments']['IfSteps'][0][
#        'Arguments']['ModelMetrics']['ModelQuality']['Statistics']['S3Uri']
#eval_report = ("s3://sagemaker-eu-central-1-799052614850"
#               "/script-bitti-eval-2021-03-31-13-52-11-985/output"
#               "/evaluation/evaluation.json")
#
## get the report from S3 with botocore, fill in the mAP score
#eval_key = eval_report.split('s3://')[-1].split('/', 1)[1]
#eval_bucket = eval_report.split('s3://')[-1].split('/')[0]
#s3 = boto3.resource('s3').Object(eval_bucket, eval_key)
#eval_data = json.load(s3.get()['Body'])

mAP = model_card_toolkit.PerformanceMetric(
        type='Mean average precision (mAP) score',
        value=eval_data['regression_metrics']['mAP']['value'])
model_card.quantitative_analysis.performance_metrics.append(mAP)

mct.update_model_card_json(model_card)
html = mct.export_format()

logging.info("Created a model card HTML: \n%s", html)
with open(OUTPUT_DIR/"model_card.html", "w") as fout:
    fout.write(html)
