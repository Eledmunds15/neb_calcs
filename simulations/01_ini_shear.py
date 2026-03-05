"""sim_framework.py

General framework for running MD dislocation simulations.
The framework automatically parses the input filename to determine:
    - Dislocation Type (Edge/Screw)
    - Line Direction (e.g., 111)
    - Glide Plane (e.g., 110 or 112)

Usage:
    mpirun -np [cores] python [script.py] --strain_velocity [v] --run_time [steps] --input [path] --potential [path]

Example - Edge Dislocation (110 plane):
    mpirun -np 16 python simulations/01_shear_run.py \\
        --strain_velocity 0.5 \\
        --run_time 30000 \\
        --input ./input/typed_Fe_E111_110_raw.lmp \\
        --potential ./potentials/ackland97.fs

Example - Screw Dislocation (112 plane):
    mpirun -np 16 python simulations/01_shear_run.py \\
        --strain_velocity 1.0 \\
        --run_time 50000 \\
        --input ./input/typed_Fe_S111_112_raw.lmp \\
        --potential ./potentials/mendelev03.fs

Output Directory Structure:
    data/[potential]_[type]_L[line]_G[glide]_v[velocity]_[time]ps_N[seed]/
"""

import argparse, re, yaml
import numpy as np
from dataclasses import dataclass, field, asdict
from pathlib import Path
from mpi4py import MPI


# ==============================================================================
# 1. Parameters
# ==============================================================================

@dataclass
class SimParams:
    # ... (previous fields) ...
    
    # Add these as fields that will be populated after initialization
    dislo_type: str = field(init=False)
    glide_plane: str = field(init=False)
    line_dir: str = field(init=False)

    def __post_init__(self):
        parts = self.input_path.stem.split('_')
        # Parts: [0]typed, [1]Fe, [2]E111, [3]110, [4]raw
        
        self.dislo_type = "Edge" if "E" in parts[2].upper() else "Screw"
        self.line_dir = parts[2][1:]
        self.glide_plane = parts[3]

    @property
    def case_name(self) -> str:
        # Now uses the clean attributes populated in post_init
        sr = f"{self.strain_velocity:.0E}".replace("+0", "").replace("+", "")
        pot = self.potential_path.stem
        return f"{pot}_{self.dislo_type}_L{self.line_dir}_G{self.glide_plane}_v{sr}"
    
# ==============================================================================
# 2. Paths
# ==============================================================================

def init_paths(params: SimParams) -> dict[str, Path]:
    """Build output paths and create the case directory."""
    base = Path("data") / params.case_name

    paths = {
        "base":     base,
        "metadata": base / "metadata.yaml",
        "logs":     base / "logs",
        "dump":     base / "dump",
        "output":   base / "output"
    }

    for p in paths.values():
        # Create directories if they don't have a file extension
        if not p.suffix:
            p.mkdir(parents=True, exist_ok=True)

    return paths

# ==============================================================================
# 3. Metadata (remains largely the same)
# ==============================================================================

def save_metadata(params: SimParams, paths: dict[str, Path]) -> None:
    metadata = asdict(params)
    for key, value in metadata.items():
        if isinstance(value, Path):
            metadata[key] = str(value)

    with open(paths["metadata"], "w") as f:
        yaml.safe_dump(metadata, f, default_flow_style=False, sort_keys=False)

# ==============================================================================
# 4. Argument Parser
# ==============================================================================

def parse_arguments() -> SimParams:
    parser = argparse.ArgumentParser(description="MD dislocation simulation runner")
    parser.add_argument("--strain_velocity", type=float, required=True)
    parser.add_argument("--input",           type=str,   required=True)
    parser.add_argument("--potential",       type=str,   required=True)
    parser.add_argument("--run_time",        type=int,   default=20000)
    
    args = parser.parse_args()

    return SimParams(
        strain_velocity=args.strain_velocity,
        input_path=Path(args.input).resolve(),
        potential_path=Path(args.potential).resolve(),
        run_time=args.run_time
    )


# ==============================================================================
# 5. Simulation
#    - Replace the body of this function with your LAMMPS logic
# ==============================================================================

def run_simulation(params: SimParams, paths: dict[str, Path], comm) -> None:
    """Run the LAMMPS simulation. Add your simulation logic here."""
    from lammps import lammps

    lmp = lammps(comm=comm)

    lmp.cmd.log(paths["logs"] / "log.lammps")

    lmp.cmd.units("metal")
    lmp.cmd.dimension(3)
    lmp.cmd.boundary("p", "s", "p") # Set shrink-wrapped boundaries along the y-direction
    lmp.cmd.atom_style("atomic")
    lmp.cmd.atom_modify("map", "yes")
    lmp.cmd.timestep(params.dt)

    lmp.cmd.processors("*", 2, "*")

    # ===========================
    # Define the interatomic potential
    # ===========================
    lmp.cmd.read_data(params.input)
    lmp.cmd.pair_style("eam/fs")
    lmp.cmd.pair_coeff("* *", params.potential_path, params.species, params.species, params.species)

    lmp.cmd.neighbor(2.0, "bin")
    lmp.cmd.neigh_modify("delay", 10, "check", "yes")

    # ===========================
    # Define groups
    # ===========================
    lmp.cmd.group("mobgrp", "type", "1")
    lmp.cmd.group("topgrp", "type", "2")
    lmp.cmd.group("botgrp", "type", "3")
    lmp.cmd.group("fixgrp", "union", "topgrp", "botgrp")
    lmp.cmd.group("all", "union", "fixgrp", "mobgrp")

    # ===========================
    # Computes
    # ===========================
    lmp.cmd.compute("stress", "all", "stress/atom", "NULL")
    lmp.cmd.compute("mobtemp", "mobgrp", "temp")
    lmp.cmd.compute("pe", "all", "pe/atom")
    lmp.cmd.compute("ke", "all", "ke/atom")

    lmp.cmd.compute("mobstressXX", "mobgrp", "reduce", "sum", "c_stress[1]")
    lmp.cmd.compute("mobstressYY", "mobgrp", "reduce", "sum", "c_stress[2]")
    lmp.cmd.compute("mobstressZZ", "mobgrp", "reduce", "sum", "c_stress[3]")
    lmp.cmd.compute("mobstressXY", "mobgrp", "reduce", "sum", "c_stress[4]")
    lmp.cmd.compute("mobstressYZ", "mobgrp", "reduce", "sum", "c_stress[5]")
    lmp.cmd.compute("mobstressXZ", "mobgrp", "reduce", "sum", "c_stress[6]")

    lmp.cmd.compute("mobpetot", "mobgrp", "reduce", "sum", "c_pe")
    lmp.cmd.compute("mobketot", "mobgrp", "reduce", "sum", "c_ke")

    # ===========================
    # Minimize
    # ===========================
    lmp.cmd.fix(3, "fixgrp", "setforce", "NULL", 0.0, "NULL")

    lmp.cmd.reset_timestep(0)
    lmp.cmd.thermo_style("custom", "step", "temp", "pe", "etotal", "pxx", "pyy", "pzz", "pxy", "pyz", "pxz", "lx", "ly", "lz")
    lmp.cmd.thermo(10)

    lmp.cmd.min_style("cg")
    lmp.cmd.minimize(0.0, 1e-6, 5000, 10000)
    
    # ===========================
    # Prepare for run
    # ===========================
    lmp.cmd.reset_timestep(0)
    lmp.cmd.thermo(params.thermo_freq)

    lmp.cmd.thermo_style(
        "custom", "step", "temp", "pe", "etotal", 
        "pxx", "pyy", "pzz", "pxy", "pyz", "pxz",
        "c_mobpetot", "c_mobketot", "c_mobtemp",
        "c_mobstressXX", "c_mobstressYY", "c_mobstressZZ",
        "c_mobstressXY", "c_mobstressYZ", "c_mobstressXZ"
    )

    lmp.cmd.dump("dump", "all", "custom", params.dump_freq, paths["dump"] / "dump_*.lammpstrj", 
                 "id", "type", "x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz", "c_pe", "c_ke", "c_stress[1]", "c_stress[2]", "c_stress[3]", "c_stress[4]", "c_stress[5]", "c_stress[6]"
                 )

    # ===========================
    # Shear Run
    # ===========================
    lmp.cmd.fix(1, "fixgrp", "nve")
    lmp.cmd.fix(2, "mobgrp", "nvt", "temp", params.temperature, params.temperature, params.dt*100.0)
    
    lmp.cmd.fix(3, "fixgrp", "setforce", 0.0, 0.0, 0.0)

    if params.dislo_type == "Edge":
        lmp.cmd.velocity("topgrp", "set", params.strain_velocity, 0.0, 0.0)
        lmp.cmd.velocity("botgrp", "set", -params.strain_velocity, 0.0, 0.0)
    else:
        lmp.cmd.velocity("topgrp", "set", 0.0, 0.0, params.strain_velocity)
        lmp.cmd.velocity("botgrp", "set", 0.0, 0.0, -params.strain_velocity)
    
    lmp.cmd.run(params.run_time)

    comm.Barrier()

    return None

# ==============================================================================
# 6. Main
# ==============================================================================

def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    params = parse_arguments()

    # Broadcast random seed
    params.random_seed = comm.bcast(params.random_seed, root=0)
    params.num_cores = size

    paths = init_paths(params, rank)
    comm.Barrier()

    if rank == 0:
        save_metadata(params, paths)

    run_simulation(params, paths, comm)


if __name__ == "__main__":
    main()