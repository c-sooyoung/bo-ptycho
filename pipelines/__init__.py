from .sobo import sobo_pipeline
from .test import test_pipeline

job_types = {
    'random+sobo': sobo_pipeline,
    'test': test_pipeline,
}
