# Synthetic training data generation sub-package.
#
# Provides:
#   - templates: Chinese prompt / response template library
#   - sft_generator: SFT conversation data synthesiser
#   - dpo_generator: DPO preference-pair synthesiser
#   - quality_filter: Post-generation quality filter

from .templates import TASK_TEMPLATES
from .sft_generator import SFTDataGenerator
from .dpo_generator import DPODataGenerator
from .quality_filter import QualityFilter

__all__ = [
    "TASK_TEMPLATES",
    "SFTDataGenerator",
    "DPODataGenerator",
    "QualityFilter",
]
