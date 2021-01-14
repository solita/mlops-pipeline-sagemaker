# Tips on local development

The instructions on how to set up the SageMaker Pipelines locally are still very scarce on the web, but I've successfully set up a connection to the pipeline locally using the following steps.

Background: the SageMaker Pipelines definition was created via a template in SageMaker Studio and lives on a git server in AWS CodeCommit repository.

To set up the sandbox environment locally:
1. Follow the steps to install `saml2aws` followed in a comment to AWS Sandbox intra page (remember to use an up-to-date login URL for it).
2. I added `"export AWS_DEFAULT_PROFILE=solita-sandbox"` to my rc file since I don't normally use anything else these days.
3. Try if it works with "aws s3 ls".
4. Run `pip install git-remote-codecommit`
5. Now you can clone the repo with `git clone codecommit://sagemaker-remote-pipeline-repo-name local-repo-folder`
6. To actually develop the pipeline locally, install it via `pip install -e .` from the folder you've cloned it into.
7. To make sure the setup works, try running a test on tutorial abalone data, `python get_pipeline_definition.py -n abalone --kwargs "{'region': 'region-code-of-the-pipeline'}"`, which should dump a huge JSON string to stdout with all the nitty-gritties about the pipeline definition.
