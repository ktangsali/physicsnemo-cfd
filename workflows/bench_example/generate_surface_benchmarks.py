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

from physicsnemo.cfd.bench.visualization.utils import (
    plot_design_scatter,
    plot_design_trend,
    plot_line,
)

from utils import (
    load_mapping,
    process_surface_results,
    plot_surface_results,
    save_results_to_csv,
    load_results_from_csv,
    save_vtps,
    load_vtps,
)

from physicsnemo.cfd.bench.interpolation.interpolate_mesh_to_pc import interpolate_mesh_to_pc

from physicsnemo.cfd.bench.metrics.aero_forces import compute_drag_and_lift
from physicsnemo.cfd.bench.metrics.l2_errors import (
    compute_l2_errors,
    compute_area_weighted_l2_errors,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_surface_results_prefetched(data_dict, device="cpu"):
    """Process surface results using prefetched data."""
    mesh = data_dict["mesh"]
    pc = data_dict["pc"]
    mesh_filename = data_dict["mesh_filename"]
    pc_filename = data_dict["pc_filename"]
    field_mapping = data_dict["field_mapping"]
    
    results = {}
    if pc_filename is None:
        print(f"Processing: {mesh_filename} on {device}")
    else:
        print(f"Processing: {mesh_filename}, {pc_filename} on {device}")
    
    # Fetch the run number from the filename
    run_idx = re.search(r"(\d+)(?=\D*$)", mesh_filename).group()
    results["run_idx"] = run_idx

    # compute drag and lift coefficients
    (
        results["Cd_true"],
        results["Cd_p_true"],
        results["Cd_f_true"],
        results["Cl_true"],
        results["Cl_p_true"],
        results["Cl_f_true"],
    ) = compute_drag_and_lift(
        mesh,
        pressure_field=field_mapping["p"],
        wss_field=field_mapping["wallShearStress"],
    )

    (
        results["Cd_pred"],
        results["Cd_p_pred"],
        results["Cd_f_pred"],
        results["Cl_pred"],
        results["Cl_p_pred"],
        results["Cl_f_pred"],
    ) = compute_drag_and_lift(
        mesh,
        pressure_field=field_mapping["pPred"],
        wss_field=field_mapping["wallShearStressPred"],
    )

    # compute L2 errors
    results["l2_errors"] = compute_l2_errors(
        mesh,
        [
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        [
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        dtype="cell",
    )
    results["l2_errors_area_wt"] = compute_area_weighted_l2_errors(
        mesh,
        [
            field_mapping["p"],
            field_mapping["wallShearStress"],
        ],
        [
            field_mapping["pPred"],
            field_mapping["wallShearStressPred"],
        ],
        dtype="cell",
    )

    # compute centerlines
    slice_y_0 = mesh.slice(normal="y", origin=(0, 0, 0))
    results["centerline_top"] = slice_y_0.clip(
        normal="z", origin=(0, 0, 0.4), invert=False
    )
    results["centerline_bottom"] = slice_y_0.clip(
        normal="z", origin=(0, 0, 0.4), invert=True
    )

    if pc is not None:
        # compute pc interpolations and results
        results["l2_errors_pc"] = compute_l2_errors(
            pc,
            [
                field_mapping["p"],
                field_mapping["wallShearStress"],
            ],
            [
                field_mapping["pPred"],
                field_mapping["wallShearStressPred"],
            ],
            dtype="point",
        )

        # compute drag and lift coefficients
        (
            results["Cd_true_pc"],
            results["Cd_p_true_pc"],
            results["Cd_f_true_pc"],
            results["Cl_true_pc"],
            results["Cl_p_true_pc"],
            results["Cl_f_true_pc"],
        ) = compute_drag_and_lift(
            pc,
            pressure_field=field_mapping["p"],
            wss_field=field_mapping["wallShearStress"],
            dtype="point",
        )

        (
            results["Cd_pred_pc"],
            results["Cd_p_pred_pc"],
            results["Cd_f_pred_pc"],
            results["Cl_pred_pc"],
            results["Cl_p_pred_pc"],
            results["Cl_f_pred_pc"],
        ) = compute_drag_and_lift(
            pc,
            pressure_field=field_mapping["pPred"],
            wss_field=field_mapping["wallShearStressPred"],
            dtype="point",
        )
    else:
        results["l2_errors_pc"] = None
        results["Cd_true_pc"] = None
        results["Cd_p_true_pc"] = None
        results["Cd_f_true_pc"] = None
        results["Cl_true_pc"] = None
        results["Cl_p_true_pc"] = None
        results["Cl_f_true_pc"] = None
        results["Cd_pred_pc"] = None
        results["Cd_p_pred_pc"] = None
        results["Cd_f_pred_pc"] = None
        results["Cl_pred_pc"] = None
        results["Cl_p_pred_pc"] = None
        results["Cl_f_pred_pc"] = None

    return results


class SurfaceBenchmarkDataset(Dataset):
    """Dataset for surface benchmark data loading with prefetching."""
    
    def __init__(self, filenames: List[Tuple[str, Optional[str]]], field_mapping: Dict[str, str]):
        self.filenames = filenames
        self.field_mapping = field_mapping
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        mesh_filename, pc_filename = self.filenames[idx]
        
        try:
            # Load mesh data
            mesh = pv.read(mesh_filename)
            mesh = mesh.point_data_to_cell_data()
            
            # Load point cloud data if available
            pc = None
            if pc_filename is not None:
                pc = pv.read(pc_filename)
                # Interpolate true results from mesh because PCs don't have them
                pc = interpolate_mesh_to_pc(
                    pc, mesh, [self.field_mapping["p"], self.field_mapping["wallShearStress"]]
                )
            
            return {
                "mesh": mesh,
                "pc": pc,
                "mesh_filename": mesh_filename,
                "pc_filename": pc_filename,
                "field_mapping": self.field_mapping
            }
            
        except Exception as e:
            logger.error(f"Error loading data for {mesh_filename}: {e}")
            return None


class DistributedSurfaceProcessor:
    """Distributed processor for surface benchmark computations."""
    
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
    

    
    def get_data_subset(self, dataset: SurfaceBenchmarkDataset):
        """Get the subset of data for this rank."""
        total_samples = len(dataset)
        samples_per_rank = total_samples // self.world_size
        start_idx = self.rank * samples_per_rank
        end_idx = start_idx + samples_per_rank if self.rank < self.world_size - 1 else total_samples
        return start_idx, end_idx
    
    def run_processing(self, dataset: SurfaceBenchmarkDataset):
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
                    
                result = process_surface_results_prefetched(sample, self.device)
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


def worker_process(rank: int, world_size: int, config: Dict, dataset: SurfaceBenchmarkDataset, 
                  result_queue: mp.Queue):
    """Worker process for distributed processing."""
    try:
        logger.info(f"Worker {rank}: Starting with config: {config}")
        
        processor = DistributedSurfaceProcessor(
            world_size=world_size,
            rank=rank,
            device=config["device"]
        )
        
        results = processor.run_processing(dataset)
        logger.info(f"Worker {rank}: Completed processing {len(results)} samples")
        result_queue.put((rank, results))
        
    except Exception as e:
        logger.error(f"Error in worker process {rank}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        result_queue.put((rank, []))


def run_distributed_processing(
    filenames: List[Tuple[str, Optional[str]]],
    field_mapping: Dict[str, str],
    world_size: int,
    device_type: str = "cpu"
) -> List[Dict]:
    """Run distributed processing with multiple workers."""
    
    # Setup distributed configuration
    configs = setup_distributed_processing(world_size, device_type)
    
    # Create dataset
    dataset = SurfaceBenchmarkDataset(filenames, field_mapping)
    
    # Create result queue
    result_queue = mp.Queue()
    
    # Start worker processes
    processes = []
    for rank in range(world_size):
        config = configs[rank]
        p = mp.Process(
            target=worker_process,
            args=(rank, world_size, config, dataset, result_queue)
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
        description="Compute the validation results for surface meshes with distributed processing"
    )
    parser.add_argument(
        "sim_results_dir", type=str, help="directory with surface vtp files"
    )
    parser.add_argument(
        "--pc-results-dir",
        type=str,
        help="directory with point cloud vtp files (optional)",
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
        default="surface_benchmarking",
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
            "wallShearStress": "wallShearStressMeanTrim",
            "pPred": "pMeanTrimPred",
            "wallShearStressPred": "wallShearStressMeanTrimPred",
        },
        help='mapping of field names to use for benchmarking, either as a path to a json file or a json string. Example: --field-mapping \'{"p": "pMeanTrim", "wallShearStress": "wallShearStressMeanTrim", "pPred": "pMeanTrimPred", "wallShearStressPred": "wallShearStressMeanTrimPred"}\'',
    )

    args = parser.parse_args()

    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)

    sim_mesh_results_dir = args.sim_results_dir
    pc_results_dir = args.pc_results_dir

    mesh_filenames = glob.glob(os.path.join(sim_mesh_results_dir, "*.vtp"))
    pc_filenames = []
    pc_file_map = {}
    run_idx_list = []

    if args.pc_results_dir:
        for pc_filename in os.listdir(args.pc_results_dir):
            run_idx_match = re.search(r"(\d+)(?=\D*$)", pc_filename)
            if run_idx_match:
                run_idx = run_idx_match.group()
                pc_file_map[run_idx] = os.path.join(args.pc_results_dir, pc_filename)

    # Match mesh filenames to pc filenames
    for filename in mesh_filenames:
        run_idx = re.search(r"(\d+)(?=\D*$)", filename).group()
        run_idx_list.append(run_idx)
        if run_idx in pc_file_map:
            pc_filenames.append(pc_file_map[run_idx])
        else:
            pc_filenames.append(None)

    filenames = list(zip(mesh_filenames, pc_filenames))
    
    # Check if we have any files to process
    if not filenames:
        logger.error(f"No VTP files found in {sim_mesh_results_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(filenames)} files to process")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Check if results already exist
    results_csv = os.path.join(output_dir, "results.csv")
    results_pc_csv = os.path.join(output_dir, "results_pc.csv")
    l2_errors_csv = os.path.join(output_dir, "l2_errors.csv")
    l2_errors_area_wt_csv = os.path.join(output_dir, "l2_errors_area_wt.csv")
    l2_errors_pc_csv = os.path.join(output_dir, "l2_errors_pc.csv")

    results = load_results_from_csv(results_csv)
    results_pc = load_results_from_csv(results_pc_csv)
    l2_errors = load_results_from_csv(l2_errors_csv)
    l2_errors_area_wt = load_results_from_csv(l2_errors_area_wt_csv)
    l2_errors_pc = load_results_from_csv(l2_errors_pc_csv)
    top_centerlines = load_vtps(output_dir, "top_centerline", run_idx_list)
    bottom_centerlines = load_vtps(output_dir, "bottom_centerline", run_idx_list)

    if (
        not results
        or not l2_errors
        or not l2_errors_area_wt
        or not top_centerlines
        or not bottom_centerlines
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
            filenames=filenames,
            field_mapping=args.field_mapping,
            world_size=args.world_size,
            device_type=args.device_type
        )

        # Prepare data for saving
        results = {
            "run_idx": [],
            "Cd_true": [],
            "Cd_p_true": [],
            "Cd_f_true": [],
            "Cl_true": [],
            "Cl_p_true": [],
            "Cl_f_true": [],
            "Cd_pred": [],
            "Cd_p_pred": [],
            "Cd_f_pred": [],
            "Cl_pred": [],
            "Cl_p_pred": [],
            "Cl_f_pred": [],
        }
        results_pc = {
            "run_idx": [],
            "Cd_true_pc": [],
            "Cd_p_true_pc": [],
            "Cd_f_true_pc": [],
            "Cl_true_pc": [],
            "Cl_p_true_pc": [],
            "Cl_f_true_pc": [],
            "Cd_pred_pc": [],
            "Cd_p_pred_pc": [],
            "Cd_f_pred_pc": [],
            "Cl_pred_pc": [],
            "Cl_p_pred_pc": [],
            "Cl_f_pred_pc": [],
        }
        # Check if we have any results
        if not mesh_results:
            logger.error("No results were processed. Check if input files exist and are accessible.")
            sys.exit(1)
            
        l2_errors = {"run_idx": []}
        for key in mesh_results[0]["l2_errors"].keys():
            l2_errors[key] = []
        l2_errors_area_wt = {"run_idx": []}
        for key in mesh_results[0]["l2_errors_area_wt"].keys():
            l2_errors_area_wt[key] = []

        l2_errors_pc = {"run_idx": []}
        if mesh_results[0]["l2_errors_pc"] is not None:
            for key in mesh_results[0]["l2_errors_pc"].keys():
                l2_errors_pc[key] = []
        else:
            l2_errors_pc = None

        top_centerlines = []
        bottom_centerlines = []

        for mesh_result in mesh_results:
            results["run_idx"].append(mesh_result["run_idx"])
            results["Cd_true"].append(mesh_result["Cd_true"])
            results["Cd_p_true"].append(mesh_result["Cd_p_true"])
            results["Cd_f_true"].append(mesh_result["Cd_f_true"])
            results["Cl_true"].append(mesh_result["Cl_true"])
            results["Cl_p_true"].append(mesh_result["Cl_p_true"])
            results["Cl_f_true"].append(mesh_result["Cl_f_true"])
            results["Cd_pred"].append(mesh_result["Cd_pred"])
            results["Cd_p_pred"].append(mesh_result["Cd_p_pred"])
            results["Cd_f_pred"].append(mesh_result["Cd_f_pred"])
            results["Cl_pred"].append(mesh_result["Cl_pred"])
            results["Cl_p_pred"].append(mesh_result["Cl_p_pred"])
            results["Cl_f_pred"].append(mesh_result["Cl_f_pred"])

            if args.pc_results_dir:
                results_pc["run_idx"].append(mesh_result["run_idx"])
                results_pc["Cd_true_pc"].append(mesh_result["Cd_true_pc"])
                results_pc["Cd_p_true_pc"].append(mesh_result["Cd_p_true_pc"])
                results_pc["Cd_f_true_pc"].append(mesh_result["Cd_f_true_pc"])
                results_pc["Cl_true_pc"].append(mesh_result["Cl_true_pc"])
                results_pc["Cl_p_true_pc"].append(mesh_result["Cl_p_true_pc"])
                results_pc["Cl_f_true_pc"].append(mesh_result["Cl_f_true_pc"])
                results_pc["Cd_pred_pc"].append(mesh_result["Cd_pred_pc"])
                results_pc["Cd_p_pred_pc"].append(mesh_result["Cd_p_pred_pc"])
                results_pc["Cd_f_pred_pc"].append(mesh_result["Cd_f_pred_pc"])
                results_pc["Cl_pred_pc"].append(mesh_result["Cl_pred_pc"])
                results_pc["Cl_p_pred_pc"].append(mesh_result["Cl_p_pred_pc"])
                results_pc["Cl_f_pred_pc"].append(mesh_result["Cl_f_pred_pc"])

                l2_errors_pc["run_idx"].append(mesh_result["run_idx"])

            l2_errors["run_idx"].append(mesh_result["run_idx"])
            l2_errors_area_wt["run_idx"].append(mesh_result["run_idx"])

            for key, value in mesh_result["l2_errors"].items():
                l2_errors[key].append(value)
            for key, value in mesh_result["l2_errors_area_wt"].items():
                l2_errors_area_wt[key].append(value)

            if args.pc_results_dir:
                for key, value in mesh_result["l2_errors_pc"].items():
                    l2_errors_pc[key].append(value)

            top_centerlines.append(mesh_result["centerline_top"])
            bottom_centerlines.append(mesh_result["centerline_bottom"])

        # Save results to CSV
        save_results_to_csv(results, results_csv, results.keys())
        save_results_to_csv(l2_errors, l2_errors_csv, l2_errors.keys())
        save_results_to_csv(
            l2_errors_area_wt, l2_errors_area_wt_csv, l2_errors_area_wt.keys()
        )

        if args.pc_results_dir:
            save_results_to_csv(results_pc, results_pc_csv, results_pc.keys())
            save_results_to_csv(l2_errors_pc, l2_errors_pc_csv, l2_errors_pc.keys())

        # Save centerlines
        save_vtps(top_centerlines, output_dir, "top_centerline", results["run_idx"])
        save_vtps(
            bottom_centerlines, output_dir, "bottom_centerline", results["run_idx"]
        )
    else:
        # Load mesh_results from the saved CSVs
        mesh_results = []
        for i in range(len(results["run_idx"])):
            # print(l2_errors_pc)
            mesh_result = {
                "run_idx": results["run_idx"][i],
                "Cd_true": results["Cd_true"][i],
                "Cd_p_true": results["Cd_p_true"][i],
                "Cd_f_true": results["Cd_f_true"][i],
                "Cl_true": results["Cl_true"][i],
                "Cl_p_true": results["Cl_p_true"][i],
                "Cl_f_true": results["Cl_f_true"][i],
                "Cd_pred": results["Cd_pred"][i],
                "Cd_p_pred": results["Cd_p_pred"][i],
                "Cd_f_pred": results["Cd_f_pred"][i],
                "Cl_pred": results["Cl_pred"][i],
                "Cl_p_pred": results["Cl_p_pred"][i],
                "Cl_f_pred": results["Cl_f_pred"][i],
                "l2_errors": {
                    key: l2_errors[key][i] for key in l2_errors if key != "run_idx"
                },
                "l2_errors_area_wt": {
                    key: l2_errors_area_wt[key][i]
                    for key in l2_errors_area_wt
                    if key != "run_idx"
                },
                "centerline_top": top_centerlines[i],
                "centerline_bottom": bottom_centerlines[i],
            }

            if args.pc_results_dir:
                mesh_result.update(
                    {
                        "Cd_true_pc": results_pc["Cd_true_pc"][i],
                        "Cd_p_true_pc": results_pc["Cd_p_true_pc"][i],
                        "Cd_f_true_pc": results_pc["Cd_f_true_pc"][i],
                        "Cl_true_pc": results_pc["Cl_true_pc"][i],
                        "Cl_p_true_pc": results_pc["Cl_p_true_pc"][i],
                        "Cl_f_true_pc": results_pc["Cl_f_true_pc"][i],
                        "Cd_pred_pc": results_pc["Cd_pred_pc"][i],
                        "Cd_p_pred_pc": results_pc["Cd_p_pred_pc"][i],
                        "Cd_f_pred_pc": results_pc["Cd_f_pred_pc"][i],
                        "Cl_pred_pc": results_pc["Cl_pred_pc"][i],
                        "Cl_p_pred_pc": results_pc["Cl_p_pred_pc"][i],
                        "Cl_f_pred_pc": results_pc["Cl_f_pred_pc"][i],
                        "l2_errors_pc": {
                            key: l2_errors_pc[key][i]
                            for key in l2_errors_pc
                            if key != "run_idx"
                        },
                    }
                )

            mesh_results.append(mesh_result)

    # combine results
    true_data_dict = {"Cd": [], "Cl": []}
    pred_data_dict = {"Cd": [], "Cl": []}
    idx_dict = {"Cd": [], "Cl": []}
    if args.pc_results_dir:
        true_data_dict_pc = {"Cd": [], "Cl": []}
        pred_data_dict_pc = {"Cd": [], "Cl": []}

    top_centerlines = []
    bottom_centerlines = []

    mean_l2_errors = {}
    for key in mesh_results[0]["l2_errors"].keys():
        mean_l2_errors[key] = []

    mean_area_wt_l2_errors = {}
    for key in mesh_results[0]["l2_errors_area_wt"].keys():
        mean_area_wt_l2_errors[key] = []

    if args.pc_results_dir:
        mean_l2_errors_pc = {}
        for key in mesh_results[0]["l2_errors_pc"].keys():
            mean_l2_errors_pc[key] = []
    else:
        mean_l2_errors_pc = None

    for mesh_result in mesh_results:
        true_data_dict["Cd"].append(mesh_result["Cd_true"])
        true_data_dict["Cl"].append(mesh_result["Cl_true"])
        pred_data_dict["Cd"].append(mesh_result["Cd_pred"])
        pred_data_dict["Cl"].append(mesh_result["Cl_pred"])
        idx_dict["Cd"].append(mesh_result["run_idx"])
        idx_dict["Cl"].append(mesh_result["run_idx"])
        top_centerlines.append(mesh_result["centerline_top"])
        bottom_centerlines.append(mesh_result["centerline_bottom"])

        for key, value in mesh_result["l2_errors"].items():
            mean_l2_errors[key].append(value)
        for key, value in mesh_result["l2_errors_area_wt"].items():
            mean_area_wt_l2_errors[key].append(value)
        if args.pc_results_dir:
            true_data_dict_pc["Cd"].append(mesh_result["Cd_true_pc"])
            true_data_dict_pc["Cl"].append(mesh_result["Cl_true_pc"])
            pred_data_dict_pc["Cd"].append(mesh_result["Cd_pred_pc"])
            pred_data_dict_pc["Cl"].append(mesh_result["Cl_pred_pc"])
            for key, value in mesh_result["l2_errors_pc"].items():
                mean_l2_errors_pc[key].append(value)

    for key, value in mean_l2_errors.items():
        mean_l2_errors[key] = np.mean(np.array(value))

    for key, value in mean_area_wt_l2_errors.items():
        mean_area_wt_l2_errors[key] = np.mean(np.array(value))

    if args.pc_results_dir:
        for key, value in mean_l2_errors_pc.items():
            mean_l2_errors_pc[key] = np.mean(np.array(value))

    fig = plot_design_scatter(
        true_data_dict,
        pred_data_dict,
        figsize=(15, 6),
        regression_line_kwargs={"color": "black", "linestyle": "--"},
        title_kwargs={"fontsize": 10},
    )[0]
    fig.savefig(f"./{output_dir}/design_scatter_plot.png")

    fig = plot_design_trend(
        true_data_dict,
        pred_data_dict,
        idx_dict,
        figsize=(15, 6),
        true_line_kwargs={"color": "red"},
        pred_line_kwargs={"color": "green"},
        title_kwargs={"fontsize": 10},
    )[0]
    fig.savefig(f"./{output_dir}/design_trend_plot.png")

    if args.pc_results_dir:
        fig = plot_design_scatter(
            true_data_dict,  # plot against the true data from simulation mesh
            pred_data_dict_pc,
            figsize=(15, 6),
            regression_line_kwargs={"color": "black", "linestyle": "--"},
            title_kwargs={"fontsize": 10},
        )[0]
        fig.savefig(f"./{output_dir}/design_scatter_pc_plot.png")

        fig = plot_design_trend(
            true_data_dict,  # plot against the true data from simulation mesh
            pred_data_dict_pc,
            idx_dict,
            figsize=(15, 6),
            true_line_kwargs={"color": "red"},
            pred_line_kwargs={"color": "green"},
            title_kwargs={"fontsize": 10},
        )[0]
        fig.savefig(f"./{output_dir}/design_trend_pc_plot.png")

    fig = plot_line(
        top_centerlines,
        plot_coord="x",
        field_true=args.field_mapping["p"],
        field_pred=args.field_mapping["pPred"],
        normalize_factor=(1.0 * 38.889**2) / 2,
        true_line_kwargs={"color": "red", "alpha": 1 / len(top_centerlines)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(top_centerlines)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        title_kwargs={"fontsize": 12},
        figsize=(8, 6),
        xlabel="X Coordinate",
        ylabel="p / U_ref^2",
    )
    fig.savefig(f"./{output_dir}/top_centerline.png")

    fig = plot_line(
        bottom_centerlines,
        plot_coord="x",
        field_true=args.field_mapping["p"],
        field_pred=args.field_mapping["pPred"],
        normalize_factor=(1.0 * 38.889**2) / 2,
        true_line_kwargs={"color": "red", "alpha": 1 / len(bottom_centerlines)},
        pred_line_kwargs={"color": "green", "alpha": 1 / len(bottom_centerlines)},
        mean_true_line_kwargs={"color": "red", "label": "Mean True"},
        mean_pred_line_kwargs={"color": "green", "label": "Mean Pred"},
        title_kwargs={"fontsize": 12},
        figsize=(8, 6),
        xlabel="X Coordinate",
        ylabel="p / U_ref^2",
    )
    fig.savefig(f"./{output_dir}/bottom_centerline.png")

    for key, value in mean_l2_errors.items():
        print(f"L2 Errors for {key}: {value}")

    for key, value in mean_area_wt_l2_errors.items():
        print(f"Area weighted L2 Errors for {key}: {value}")

    if args.pc_results_dir:
        for key, value in mean_l2_errors_pc.items():
            print(f"L2 Errors for PC, {key}: {value}")

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
                    plot_surface_results,
                    field_mapping=args.field_mapping,
                    output_dir=args.output_dir,
                ),
                plot_filenames,
            )
