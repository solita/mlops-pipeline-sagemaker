"""Parses config.ini into sagemaker workflow parameter instances"""
from configparser import ConfigParser, ExtendedInterpolation
import boto3
from sagemaker.workflow.parameters import (ParameterInteger, ParameterString,
                                           ParameterFloat)
from sagemaker.workflow.steps import CacheConfig


def read_conf(cfg_file):
    """Reads config file, returns a dict with workflow parameters"""

    # FIXME: refactor! the function is ugly, instead we can set the
    #        three names from one present in the .ini file. That would
    #        likely need to have some snake case to PascalCase and back
    #        conversion hacks
    cfg_dict = {}
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read_file(open(cfg_file))

    region = boto3.Session().region_name
    cfg_dict['bucket'] = config['metadata'].get('bucket')

    # fetch the execution role and account id from Secrets Manager
    session = boto3.session.Session()
    client_secrets_fetch = session.client(
        service_name='secretsmanager',
        region_name=region
    ).get_secret_value

    secret_role = config['secretsmanager'].get('secret_role')
    secret_account_id = config['secretsmanager'].get('secret_account_id')
    cfg_dict['role'] = client_secrets_fetch(SecretId=secret_role)['SecretString']
    account_id = client_secrets_fetch(
            SecretId=secret_account_id)['SecretString']

    # will reuse the same cache config for all the steps
    cfg_dict['cache_config'] = CacheConfig(
            enable_caching=config['metadata'].getboolean('cache_steps'),
            expire_after=config['metadata'].get('cache_expire_after'))
    # FIXME: resolve with pathlib!
    cfg_dict['source_dir'] = config['metadata'].get('source_dir')

    # start defining workflow parameters for sagemaker pipeline steps
    # first off, processing steps
    cfg_dict['input_data'] = ParameterString(
        name='InputData',
        default_value=config['processing'].get('input_data'))

    cfg_dict['processing_instance_count'] = ParameterInteger(
        name='ProcessingInstanceCount',
        default_value=config['processing'].getint('instance_count'))

    cfg_dict['processing_instance_type'] = ParameterString(
        name='ProcessingInstanceType',
        default_value=config['processing'].get('instance_type'))

    cfg_dict['processing_train_test_split'] = ParameterFloat(
        name='TrainTestSplit',
        default_value=config['processing'].getfloat('train_test_split_fraction'))

    cfg_dict['processing_turicreate_uri'] = ParameterString(
        name='TuriCreateProcessingURI',
        default_value=config['processing'].get('image_uri_fmt').format(account_id))

    # control settings for the training job
    cfg_dict['training_instance_count'] = ParameterInteger(
        name='TrainingInstanceCount',
        default_value=config['training'].getint('instance_count'))

    cfg_dict['training_instance_type'] = ParameterString(
        name='TrainingInstanceType',
        default_value=config['training'].get('instance_type'))

    cfg_dict['training_batch_size'] = ParameterInteger(
        name='TrainingBatchSize',
        default_value=config['training'].getint('batch_size'))

    cfg_dict['training_max_iterations'] = ParameterInteger(
        name='MaxIterations',
        default_value=config['training'].getint('max_iterations'))

    cfg_dict['training_turicreate_uri'] = ParameterString(
        name='TuriCreateTrainingURI',
        default_value=config['training'].get('image_uri_fmt').format(account_id))

    # workflow parameters for model approval / rejection
    cfg_dict['model_approval_status'] = ParameterString(
        name='ModelApprovalStatus',
        default_value=config['evaluation'].get('approval_status'))

    cfg_dict['model_approval_map_threshold'] = ParameterFloat(
        name='ModelApprovalmAPThreshold',
        default_value=config['evaluation'].getfloat('approval_map_threshold'))

    cfg_dict['model_package_group_name'] = config['metadata'].get(
            'model_package_group_name')
    return cfg_dict
