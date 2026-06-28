"""
src/config/__init__.py

Configuration management for local and cloud environments.
"""

from .env_config import (
    IS_KAGGLE,
    IS_COLAB,
    IS_LOCAL,
    ENVIRONMENT,
    PYTHON_VERSION,
    PROJECT_ROOT,
    get_test_video_dir,
    get_output_dir,
    get_checkpoint_dir,
    get_config_dir,
    print_environment_info,
)

__all__ = [
    "IS_KAGGLE",
    "IS_COLAB",
    "IS_LOCAL",
    "ENVIRONMENT",
    "PYTHON_VERSION",
    "PROJECT_ROOT",
    "get_test_video_dir",
    "get_output_dir",
    "get_checkpoint_dir",
    "get_config_dir",
    "print_environment_info",
]
