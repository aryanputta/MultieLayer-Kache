"""
setup.py — Adaptive Multi-Tier KV Cache Orchestrator for Long-Context LLM Inference
Package: kv_orchestrator  |  Version: 0.1.0
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

# ---------------------------------------------------------------------------
# Long description from README (optional — graceful fallback)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_README = _HERE / "README.md"
long_description = _README.read_text(encoding="utf-8") if _README.exists() else ""

# ---------------------------------------------------------------------------
# Shared runtime requirements (mirrors requirements.txt core deps)
# ---------------------------------------------------------------------------
INSTALL_REQUIRES = [
    "vllm>=0.4.0,<0.7.0",
    "torch>=2.1.0,<2.5.0",
    "transformers>=4.40.0,<5.0.0",
    "accelerate>=0.27.0,<1.0.0",
    "fastapi>=0.110.0,<1.0.0",
    "uvicorn[standard]>=0.27.0,<1.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "aiofiles>=23.2.0,<25.0.0",
    "httpx>=0.27.0,<1.0.0",
    "ray>=2.10.0,<3.0.0",
    "numpy>=1.26.0,<2.1.0",
    "pandas>=2.2.0,<3.0.0",
    "pyarrow>=14.0.0,<17.0.0",
    "scipy>=1.12.0,<2.0.0",
    "scikit-learn>=1.4.0,<2.0.0",
    "xgboost>=2.0.0,<3.0.0",
    "lightgbm>=4.3.0,<5.0.0",
    "optuna>=3.6.0,<4.0.0",
    "hydra-core>=1.3.0,<2.0.0",
    "omegaconf>=2.3.0,<3.0.0",
    "prometheus-client>=0.20.0,<1.0.0",
    "psutil>=5.9.0,<7.0.0",
    "gputil>=1.4.0",
    "pynvml>=11.5.0,<12.0.0",
    "plotly>=5.20.0,<6.0.0",
    "matplotlib>=3.8.0,<4.0.0",
    "seaborn>=0.13.0,<1.0.0",
    "datasets>=2.18.0,<4.0.0",
    "evaluate>=0.4.0,<1.0.0",
    "rouge-score>=0.1.2,<1.0.0",
    "nltk>=3.8.0,<4.0.0",
    "redis>=5.0.0,<6.0.0",
    "tqdm>=4.66.0,<5.0.0",
    "rich>=13.7.0,<14.0.0",
    "typer>=0.12.0,<1.0.0",
    "mlflow>=2.11.0,<3.0.0",
    "wandb>=0.16.0,<1.0.0",
]

EXTRAS_REQUIRE = {
    # Full development toolkit
    "dev": [
        "pytest>=8.0.0,<9.0.0",
        "pytest-asyncio>=0.23.0,<1.0.0",
        "pytest-cov>=4.1.0,<6.0.0",
        "pytest-mock>=3.12.0,<4.0.0",
        "ruff>=0.4.0",
        "mypy>=1.9.0",
        "pre-commit>=3.7.0",
        "ipython>=8.23.0",
        "ipykernel>=6.29.0",
    ],
    # Minimal test-only install (CI)
    "test": [
        "pytest>=8.0.0,<9.0.0",
        "pytest-asyncio>=0.23.0,<1.0.0",
        "pytest-cov>=4.1.0,<6.0.0",
        "pytest-mock>=3.12.0,<4.0.0",
        "httpx>=0.27.0,<1.0.0",
    ],
    # Optional: NVMe offload acceleration
    "nvme": [
        "deepspeed>=0.14.0",
    ],
}

# Convenience "all" extras
EXTRAS_REQUIRE["all"] = sorted(
    {dep for extra_deps in EXTRAS_REQUIRE.values() for dep in extra_deps}
)

setup(
    # -----------------------------------------------------------------------
    # Identity
    # -----------------------------------------------------------------------
    name="kv_orchestrator",
    version="0.1.0",
    description=(
        "Adaptive Multi-Tier KV Cache Orchestrator for Long-Context LLM Inference"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MultieLayer-Kache Contributors",
    author_email="",
    url="https://github.com/your-org/MultieLayer-Kache",
    license="Apache-2.0",
    # -----------------------------------------------------------------------
    # Python / package discovery
    # -----------------------------------------------------------------------
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    # -----------------------------------------------------------------------
    # Entry points (CLI)
    # -----------------------------------------------------------------------
    entry_points={
        "console_scripts": [
            # Main orchestrator daemon
            "kv-orchestrator=kv_orchestrator.cli:app",
            # Benchmark runner
            "kv-benchmark=kv_orchestrator.benchmark.cli:app",
        ],
    },
    # -----------------------------------------------------------------------
    # Packaging metadata
    # -----------------------------------------------------------------------
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: POSIX :: Linux",
    ],
    keywords=[
        "llm",
        "kv-cache",
        "vllm",
        "inference",
        "long-context",
        "memory-management",
        "machine-learning",
        "ray",
    ],
)
