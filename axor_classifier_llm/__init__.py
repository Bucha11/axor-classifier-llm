from axor_classifier_llm._version import get_version
from axor_classifier_llm.verifier import LLMAnomalyVerifier

__version__ = get_version("axor-classifier-llm")

__all__ = ["LLMAnomalyVerifier", "__version__"]
