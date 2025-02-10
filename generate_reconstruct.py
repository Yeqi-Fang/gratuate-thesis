# generate_reconstruct.py

import os
import re
import glob
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from pytomography.metadata import ObjectMeta
from pytomography.metadata.PET import PETLMProjMeta
from pytomography.projectors.PET import PETLMSystemMatrix
from pytomography.algorithms import OSEM
from pytomography.likelihoods import PoissonLogLikelihood
from pytomography.transforms.shared import GaussianFilter

from pet_simulator.geometry import create_pet_geometry
from pet_simulator.simulator import PETSimulator
from pet_simulator.utils import save_events, save_events_binary

from outlier_detection import (
    global_outlier_detection,
    local_outlier_detection,
    edge_outlier_detection,
    combined_outlier_detection,
    analyze_outlier_masks,
    remove_outliers_iteratively
)

