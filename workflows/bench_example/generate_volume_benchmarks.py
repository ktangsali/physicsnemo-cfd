# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os
import pyvista as pv
import glob
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
import argparse
import json
from functools import partial
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import queue
import threading
import time
from typing import List, Tuple, Dict, Optional, Union
import logging
import warnings
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()

from physicsnemo.cfd.bench.visualization.utils import plot_line
from utils import (
    load_mapping,
    process_volume_results,
    plot_volume_results,
    save_results_to_csv,
    load_results_from_csv,
    save_vtps,
    load_vtps,
)

from physicsnemo.cfd.bench.metrics.l2_errors import compute_l2_errors
from physicsnemo.cfd.bench.metrics.physics import (
    compute_continuity_residuals,
    compute_momentum_residuals,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_volume_results_prefetched(data_dict, device="cpu", compute_continuity_metrics=False, compute_momentum_metrics=False, nu=None, rho=None):
    """Process volume results using prefetched data."""
    mesh = data_dict["mesh"]
    filename = data_dict["filename"]
    field_mapping = data_dict["field_mapping"]
    
    results = {}
    print(f"Processing: {filename} on {device}")
    
    # Fetch the run number from the filename
    run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()
    results["run_idx"] = run_idx

    l2_errors_true_fields = [
        field_mapping["p"],
        field_mapping["U"],
        field_mapping["nut"],
    ]
    l2_errors_pred_fields = [
        field_mapping["pPred"],
        field_mapping["UPred"],
        field_mapping["nutPred"],
    ]

    if compute_continuity_metrics:
        mesh = compute_continuity_residuals(
            mesh,
            true_velocity_field=field_mapping["U"],
            predicted_velocity_field=field_mapping["UPred"],
            device=device,
        )
        l2_errors_true_fields.extend(
            [
                "Continuity",
            ]
        )
        l2_errors_pred_fields.extend(
            [
                "ContinuityPred",
            ]
        )

    if compute_momentum_metrics:
        if nu is None:
            # nu = 1.5881327800829875e-5
            nu = 1.507e-5
            warnings.warn(f"nu is not provided. Defaulting to {nu}")
        if rho is None:
            # rho = 1.225
            rho = 1.0
            warnings.warn(f"rho is not provided. Defaulting to {rho}")

        mesh = compute_momentum_residuals(
            mesh,
            true_velocity_field=field_mapping["U"],
            predicted_velocity_field=field_mapping["UPred"],
            true_pressure_field=field_mapping["p"],
            predicted_pressure_field=field_mapping["pPred"],
            true_nu_field=field_mapping["nut"],
            predicted_nu_field=field_mapping["nutPred"],
            nu=nu,
            rho=rho,
            device=device,
        )
        l2_errors_true_fields.extend(
            [
                "Momentum",
            ]
        )
        l2_errors_pred_fields.extend(["MomentumPred"])

    results["l2_errors"] = compute_l2_errors(
        mesh,
        l2_errors_true_fields,
        l2_errors_pred_fields,
        bounds=[-3.5, 8.5, -2.25, 2.25, -0.32, 3.00],
        dtype="point",
    )

    # compute lines
    y_slice = mesh.slice(normal="y", origin=(0, 0, 0))
    results["centerline_bottom"] = y_slice.slice(normal="z", origin=(0, 0, -0.2376))
    x_slice = mesh.slice(normal="x", origin=(0.35, 0, 0))
    results["front_wheel_wake"] = x_slice.slice(normal="z", origin=(0, 0, -0.2376))
    x_slice = mesh.slice(normal="x", origin=(3.15, 0, 0))
    results["rear_wheel_wake"] = x_slice.slice(normal="z", origin=(0, 0, -0.2376))
    x_slice = mesh.slice(normal="x", origin=(4, 0, 0))
    results["wake_x_4"] = x_slice.slice(normal="y", origin=(0, 0, 0))
    x_slice = mesh.slice(normal="x", origin=(5, 0, 0))
    results["wake_x_5"] = x_slice.slice(normal="y", origin=(0, 0, 0))

    return results


class VolumeBenchmarkDataset(Dataset):
    """Dataset for volume benchmark data loading with prefetching."""
    
    def __init__(self, filenames: List[str], field_mapping: Dict[str, str]):
        self.filenames = filenames
        self.field_mapping = field_mapping
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        try:
            # Load mesh data
            mesh = pv.read(filename)
            
            return {
                "mesh": mesh,
                "filename": filename,
                "field_mapping": self.field_mapping
            }
            
        except Exception as e:
            logger.error(f"Error loading data for {filename}: {e}")
            return None


class DistributedVolumeProcessor:
    """Distributed processor for volume benchmark computations."""
    
    def __init__(
        self,
        world_size: int,
        rank: int,
        device: str = "cpu"
    ):
        self.world_size = world_size
        self.rank = rank
        self.device = device
        
        # Set device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            else:
                torch.cuda.set_device(int(device.split(":")[1]) if ":" in device else 0)
        
        logger.info(f"Initialized processor: rank={rank}, device={self.device}")
    
    def get_data_subset(self, dataset: VolumeBenchmarkDataset):
        """Get the subset of data for this rank."""
        total_samples = len(dataset)
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if self.rank < self.world_size - 1 else total_samples
        return start_idx, end_idx
    
    def run_processing(self, dataset: VolumeBenchmarkDataset, compute_continuity_metrics=False, compute_momentum_metrics=False, nu=None, rho=None):
        """Run distributed processing."""
        # Get the subset of data for this rank
        total_samples = len(dataset)
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if self.rank < self.world_size - 1 else total_samples
        
        # Process samples one by one
        all_results = []
        for i in range(start_idx, end_idx):
            sample = dataset[i]
            logger.info(f"Rank {self.rank}: Processing sample {i+1}/{total_samples}")
            
            # Process single sample
            try:
                if sample is None:
                    logger.warning(f"Rank {self.rank}: Skipping None sample at index {i}")
                    continue
                    
                result = process_volume_results_prefetched(
                    sample, 
                    self.device, 
                    compute_continuity_metrics, 
                    compute_momentum_metrics, 
                    nu, 
                    rho
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Rank {self.rank}: Error processing sample {i}: {e}")
                continue
            
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
        
        logger.info(f"Rank {self.rank}: Completed processing {len(all_results)} samples")
        return all_results


def setup_distributed_processing(
    world_size: int,
    device_type: str = "cpu"
) -> List[Dict]:
    """Setup distributed processing configuration."""
    configs = []
    
    if device_type == "gpu":
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if available_gpus < world_size:
            logger.warning(f"Requested {world_size} GPUs but only {available_gpus} available. Some processes will share GPUs.")
        
        for rank in range(world_size):
            gpu_id = rank % max(available_gpus, 1)
            configs.append({
                "rank": rank,
                "device": f"cuda:{gpu_id}"
            })
    else:
        for rank in range(world_size):
            configs.append({
                "rank": rank,
                "device": "cpu"
            })
    
    return configs


def worker_process(rank: int, world_size: int, config: Dict, dataset: VolumeBenchmarkDataset, 
                  result_queue: mp.Queue, compute_continuity_metrics=False, compute_momentum_metrics=False, nu=None, rho=None):
    """Worker process for distributed processing."""
    try:
        logger.info(f"Worker {rank}: Starting with config: {config}")
        
        processor = DistributedVolumeProcessor(
            world_size=world_size,
            rank=rank,
            device=config["device"]
        )
        
        results = processor.run_processing(dataset, compute_continuity_metrics, compute_momentum_metrics, nu, rho)
        logger.info(f"Worker {rank}: Completed processing {len(results)} samples")
        result_queue.put((rank, results))
        
    except Exception as e:
        logger.error(f"Error in worker process {rank}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        result_queue.put((rank, []))


def run_distributed_processing(
    filenames: List[str],
    field_mapping: Dict[str, str],
    world_size: int,
    device_type: str = "cpu",
    compute_continuity_metrics=False,
    compute_momentum_metrics=False,
    nu=None,
    rho=None
) -> List[Dict]:
    """Run distributed processing with multiple workers."""
    
    # Setup distributed configuration
    configs = setup_distributed_processing(world_size, device_type)
    
    # Create dataset
    dataset = VolumeBenchmarkDataset(filenames, field_mapping)
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for rank in range(world_size):
        config = configs[rank]
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, config, dataset, result_queue, compute_continuity_metrics, compute_momentum_metrics, nu, rho)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    all_results = []
    for _ in range(world_size):
        rank, results = result_queue.get()
        all_results.extend(results)
        logger.info(f"Received results from rank {rank}: {len(results)} samples")
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the validation results for volume meshes with distributed processing"
    )
    parser.add_argument(
        "sim_results_dir", type=str, help="directory with volume vtu files"
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="number of parallel processes to use",
    )
    parser.add_argument(
        "--device-type",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="device type for processing (cpu or gpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="volume_benchmarking",
        help="directory to store results",
    )
    parser.add_argument(
        "--contour-plot-ids",
        nargs="+",
        type=str,
        help="run indices to plot contour plots for",
    )
    parser.add_argument(
        "--field-mapping",
        type=load_mapping,
        default={
            "p": "pMeanTrim",
            "U": "UMeanTrim",
            "nut": "nutMeanTrim",
            "pPred": "pMeanTrimPred",
            "UPred": "UMeanTrimPred",
            "nutPred": "nutMeanTrimPred",
        },
        help='mapping of field names to use for benchmarking, either as a path to a json file or a json string. Example: --field-mapping \'{"p": "pMeanTrim", "wallShearStress": "wallShearStressMeanTrim", "pPred": "pMeanTrimPred", "wallShearStressPred": "wallShearStressMeanTrimPred"}\'',
    )

    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    compute_continuity_metrics = True
    compute_momentum_metrics = False
    plot_continuity_metrics = True
    plot_momentum_metrics = True
    sim_mesh_results_dir = args.sim_results_dir

    mesh_filenames = glob.glob(os.path.join(sim_mesh_results_dir, "*.vtu"))
    
    # Check if we have any files to process
    if not mesh_filenames:
        logger.error(f"No VTU files found in {sim_mesh_results_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(mesh_filenames)} files to process")
    
    run_idx_list = []
    for filename in mesh_filenames:
        run_idx_match = re.search(r"(\d+)(?=\D*$)", filename)
        if run_idx_match:
            run_idx = run_idx_match.group()
            run_idx_list.append(run_idx)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if results already exist
    l2_errors_csv = os.path.join(output_dir, "l2_errors.csv")
    l2_errors = load_results_from_csv(l2_errors_csv)

    centerlines_bottom = load_vtps(output_dir, "centerline_bottom", run_idx_list)
    front_wheel_wakes = load_vtps(output_dir, "front_wheel_wake", run_idx_list)
    rear_wheel_wakes = load_vtps(output_dir, "rear_wheel_wake", run_idx_list)
    wakes_x_4 = load_vtps(output_dir, "wake_x_4", run_idx_list)
    wakes_x_5 = load_vtps(output_dir, "wake_x_5", run_idx_list)

    if (
        not l2_errors
        or not centerlines_bottom
        or not front_wheel_wakes
        or not rear_wheel_wakes
        or not wakes_x_4
        or not wakes_x_5
    ):
        # Process the data using distributed processing
        logger.info(f"Starting distributed processing with {args.world_size} processes")
        logger.info(f"Device type: {args.device_type}")
        if args.device_type == "gpu":
            available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            logger.info(f"Available GPUs: {available_gpus}")
            if available_gpus < args.world_size:
                logger.warning(f"Requested {args.world_size} processes but only {available_gpus} GPUs available")
        
        mesh_results = run_distributed_processing(
            filenames=mesh_filenames,
            field_mapping=args.field_mapping,
            world_size=args.world_size,
            device_type=args.device_type,
            compute_continuity_metrics=compute_continuity_metrics,
            compute_momentum_metrics=compute_momentum_metrics
        )

        # Check if we have any results
        if not mesh_results:
            logger.error("No results were processed. Check if input files exist and are accessible.")
            sys.exit(1)

        # Prepare data for saving
        l2_errors = {"run_idx": []}
        for key in mesh_results[0]["l2_errors"].keys():
            l2_errors[key] = []

        centerlines_bottom = []
        front_wheel_wakes = []
        rear_wheel_wakes = []
        wakes_x_4 = []
        wakes_x_5 = []

        for mesh_result in mesh_results:
            l2_errors["run_idx"].append(mesh_result["run_idx"])
            for key, value in mesh_result["l2_errors"].items():
                l2_errors[key].append(value)

            centerlines_bottom.append(mesh_result["centerline_bottom"])
            front_wheel_wakes.append(mesh_result["front_wheel_wake"])
            rear_wheel_wakes.append(mesh_result["rear_wheel_wake"])
            wakes_x_4.append(mesh_result["wake_x_4"])
            wakes_x_5.append(mesh_result["wake_x_5"])

        # Save results to CSV
        save_results_to_csv(l2_errors, l2_errors_csv, l2_errors.keys())

        # Save vtps
        save_vtps(
            centerlines_bottom, output_dir, "centerline_bottom", l2_errors["run_idx"]
        )
        save_vtps(
            front_wheel_wakes, output_dir, "front_wheel_wake", l2_errors["run_idx"]
        )
        save_vtps(rear_wheel_wakes, output_dir, "rear_wheel_wake", l2_errors["run_idx"])
        save_vtps(wakes_x_4, output_dir, "wake_x_4", l2_errors["run_idx"])
        save_vtps(wakes_x_5, output_dir, "wake_x_5", l2_errors["run_idx"])

    else:
        # Load results from saved CSVs
        mesh_results = []
        for i in range(len(l2_errors["run_idx"])):
            mesh_result = {
                "run_idx": l2_errors["run_idx"][i],
                "l2_errors": {
                    key: l2_errors[key][i] for key in l2_errors if key != "run_idx"
                },
                "centerline_bottom": centerlines_bottom[i],
                "front_wheel_wake": front_wheel_wakes[i],
                "rear_wheel_wake": rear_wheel_wakes[i],
                "wake_x_4": wakes_x_4[i],
                "wake_x_5": wakes_x_5[i],
            }
            mesh_results.append(mesh_result)

    centerlines_bottom = []
    front_wheel_wakes = []
    rear_wheel_wakes = []
    wakes_x_4 = []
    wakes_x_5 = []
    mean_l2_errors = {}

    for key in mesh_results[0]["l2_errors"].keys():
        mean_l2_errors[key] = []

    for mesh_result in mesh_results:
        centerlines_bottom.append(mesh_result["centerline_bottom"])
        front_wheel_wakes.append(mesh_result["front_wheel_wake"])
        rear_wheel_wakes.append(mesh_result["rear_wheel_wake"])
        wakes_x_4.append(mesh_result["wake_x_4"])
        wakes_x_5.append(mesh_result["wake_x_5"])

        for key, value in mesh_result["l2_errors"].items():
            mean_l2_errors[key].append(value)

    for key, values in mean_l2_errors.items():
        mean_l2_errors[key] = np.mean(np.array(values))

    for key, value in mean_l2_errors.items():
        print(f"L2 Errors for {key}: {value}")

    fig = plot_line(
        centerlines_bottom,
        plot_coord="x",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-1.0, 6.0),
        field_trim=(0, 2.0),
        flip=False,
        true_line_kwargs={"color": "red", "alpha": 1 / len(centerlines_bottom)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(centerlines_bottom)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="X Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_centerline.png")

    fig = plot_line(
        front_wheel_wakes,
        plot_coord="y",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-1.0, 1.0),
        field_trim=(0, 2.0),
        flip=False,
        true_line_kwargs={"color": "red", "alpha": 1 / len(front_wheel_wakes)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(front_wheel_wakes)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Y Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_front_wheel_wake.png")

    fig = plot_line(
        rear_wheel_wakes,
        plot_coord="y",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-1.0, 1.0),
        field_trim=(0, 2.0),
        flip=False,
        true_line_kwargs={"color": "red", "alpha": 1 / len(rear_wheel_wakes)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(rear_wheel_wakes)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Y Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_rear_wheel_wake.png")

    fig = plot_line(
        wakes_x_4,
        plot_coord="z",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-0.5, 1.5),
        field_trim=(0, 2.0),
        flip=True,
        true_line_kwargs={"color": "red", "alpha": 1 / len(wakes_x_4)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(wakes_x_4)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Z Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_x_4_wake.png")

    fig = plot_line(
        wakes_x_5,
        plot_coord="z",
        field_true=args.field_mapping["U"],
        field_pred=args.field_mapping["UPred"],
        normalize_factor=38.889,
        coord_trim=(-0.5, 1.5),
        field_trim=(0, 2.0),
        flip=True,
        true_line_kwargs={"color": "red", "alpha": 1 / len(wakes_x_5)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(wakes_x_5)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        xlabel="Z Coordinate",
        ylabel="U / U_ref",
    )
    fig.savefig(f"./{output_dir}/volume_x_5_wake.png")

    if args.contour_plot_ids is not None:
        plot_filenames = []
        for filename in mesh_filenames:
            run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()
    
            if run_idx in args.contour_plot_ids:
                plot_filenames.append(filename)
    
        print(f"Plotting contour plots for {args.contour_plot_ids}")
        from multiprocessing import Pool
        with Pool(processes=args.world_size) as pool:
            _ = pool.map(
                partial(
                    plot_volume_results,
                    field_mapping=args.field_mapping,
                    output_dir=args.output_dir,
                    compute_continuity_metrics=plot_continuity_metrics,
                    compute_momentum_metrics=plot_momentum_metrics,
                ),
                plot_filenames,
            )
