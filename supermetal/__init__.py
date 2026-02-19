"""
SuperMetal: Metal Ion Location Prediction in Proteins
======================================================

A generative AI framework for predicting metal ion binding sites in protein 
structures using 3D diffusion-based modeling.

Quick Start:
    >>> import supermetal
    >>> results = supermetal.predict("protein.pdb")
    >>> print(results['positions'])

Installation:
    pip install -e .

For more information, see the project README.md
"""

from predict import predict, load_model, save_predictions_to_pdb

__version__ = "0.1.0"
__author__ = "Xiaobo Lin"

__all__ = [
    "predict",
    "load_model", 
    "save_predictions_to_pdb",
]
