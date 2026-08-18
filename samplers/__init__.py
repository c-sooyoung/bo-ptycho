from .base import Sampler
from .random import RandomSampler
from .sobo import SOBOSampler

samplers = {
    'sobo':   SOBOSampler,
    'random': RandomSampler,
}