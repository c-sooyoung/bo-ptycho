# from .sobo import sobo_pipeline # depreacated
from .test import test_pipeline
from .batched_sobo import sobo_pipeline

job_types = {
    'random+sobo': sobo_pipeline,
    'test': test_pipeline,
}
