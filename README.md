# Object Detection with Amazon AWS SageMaker Pipelines and Core ML

Logo detection example running on a small set of custom-collected data. Grabs YOLO-annotations and images, repackages them into SFrames, runs and validates Turi Create training job on SageMaker and pushes the validated `.mlmodel` to an REST API that handles file downloads.

In addition, a dedicated pipeline step generates a [model card](https://modelcards.withgoogle.com/about) for the purposes for [AIGA](https://des.utu.fi/projects/aiga/) WP3.
