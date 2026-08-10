from .example import ExamplePtychoEngine
from .fold_slice import FoldSlicePtychoEngine

engines = {
    'fake':       ExamplePtychoEngine,
    'fold_slice': FoldSlicePtychoEngine
}