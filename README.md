# Object Detection with Amazon AWS SageMaker Pipelines and Core ML

Logo detection example running on a small set of custom-collected data. Grabs YOLO-annotations and images, repackages them into SFrames, runs and validates Turi Create training job on SageMaker and pushes the validated `.mlmodel` to an REST API that handles file downloads.

In addition, a dedicated pipeline step generates a [model card](https://modelcards.withgoogle.com/about) for the purposes for [AIGA](https://des.utu.fi/projects/aiga/) WP3. A publicly available model card automatically generated from the last model execution is available on this [static page](http://aigamodelcards.com.s3-website.eu-central-1.amazonaws.com/).

## How to use and adapt for custom projects

Currently, the code assumes that all non-SageMaker resources - Lambda functions, S3 buckets and training data in them - are already provisioned on AWS.

The pipeline can be updated and run as follows, from under the pipelines folder (TODO: add a setup.py file):

```python
from pipeline import pipeline, conf

# updates the pipeline definition on SageMaker
pipeline.upsert(role_arn=conf.role)

# starts a new pipeline execution
pipeline.start()
```

To adapt to your own needs, the step functions - processing and training need to be replaced with the ones that best suit your model training. So does the `config.ini` file in the `pipelines` folder, which is a convenience shortcut for configuration variables.
