"""
Calculate ligand RMSD in ChimeraX using optimal atom matching by element.

The reference model is fixed (default: #1). Every other atomic model opened in
ChimeraX (#2, #3, #4, ...) is compared with the reference in a single run.

Usage:
  1. Open the experimental/reference ligand as model #1.
  2. Open all docked poses as models #2, #3, #4, ...
  3. Open/run this .py file in ChimeraX.
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment


# Configuration
REFERENCE_MODEL_ID = 1
REMOVE_H = True
PRINT_ASSIGNMENT = False  # True prints atom-by-atom matching for every pose.


def get_heavy_atoms(model, remove_h=True):
    """Return model atoms, optionally excluding hydrogens."""
    atoms = list(model.atoms)
    if remove_h:
        atoms = [atom for atom in atoms if atom.element.name != "H"]
    return atoms


def get_atomic_models(session):
    """Return top-level atomic models indexed by their primary ChimeraX ID."""
    models = {}
    for model in session.models:
        if not hasattr(model, "atoms") or not model.id:
            continue
        model_id = model.id[0]
        # Prefer a true top-level model when duplicate primary IDs occur.
        if model_id not in models or len(model.id) < len(models[model_id].id):
            models[model_id] = model
    return models


def calc_pair_rmsd(reference_model, pose_model, reference_id, pose_id,
                   remove_h=REMOVE_H, print_assignment=PRINT_ASSIGNMENT):
    """Calculate RMSD between one pose and the fixed reference model."""
    atoms_ref = get_heavy_atoms(reference_model, remove_h)
    atoms_pose = get_heavy_atoms(pose_model, remove_h)

    print(f"\n{'=' * 68}")
    print(f"Reference #{reference_id}: {reference_model.name} -> {len(atoms_ref)} heavy atoms")
    print(f"Pose      #{pose_id}: {pose_model.name} -> {len(atoms_pose)} heavy atoms")

    if len(atoms_ref) != len(atoms_pose):
        message = (
            f"Different atom counts ({len(atoms_ref)} vs {len(atoms_pose)}). "
            "Check whether both models contain the same molecule."
        )
        print(f"ERROR: {message}")
        return None, message

    elements_ref = Counter(atom.element.name for atom in atoms_ref)
    elements_pose = Counter(atom.element.name for atom in atoms_pose)
    if elements_ref != elements_pose:
        message = (
            f"Different elemental composition: #{reference_id}={dict(elements_ref)}, "
            f"#{pose_id}={dict(elements_pose)}"
        )
        print(f"ERROR: {message}")
        return None, message

    coords_ref = np.array([atom.coord for atom in atoms_ref], dtype=float)
    coords_pose = np.array([atom.coord for atom in atoms_pose], dtype=float)

    n_atoms = len(atoms_ref)
    cost = np.empty((n_atoms, n_atoms), dtype=float)

    # Hungarian assignment: spatial distance plus a prohibitive element mismatch.
    for i in range(n_atoms):
        for j in range(n_atoms):
            distance = np.linalg.norm(coords_ref[i] - coords_pose[j])
            penalty = 0.0 if atoms_ref[i].element.name == atoms_pose[j].element.name else 1000.0
            cost[i, j] = distance + penalty

    row_ind, col_ind = linear_sum_assignment(cost)

    mismatches = [
        (r, c)
        for r, c in zip(row_ind, col_ind)
        if atoms_ref[r].element.name != atoms_pose[c].element.name
    ]
    if mismatches:
        message = f"Optimal assignment contains {len(mismatches)} element mismatches."
        print(f"ERROR: {message}")
        return None, message

    if print_assignment:
        print("\nOptimal assignment (Hungarian algorithm):")
        for r, c in zip(row_ind, col_ind):
            atom_ref = atoms_ref[r]
            atom_pose = atoms_pose[c]
            distance = np.linalg.norm(coords_ref[r] - coords_pose[c])
            print(
                f"  #{reference_id}:{atom_ref.residue.name}{atom_ref.residue.number}@{atom_ref.name} "
                f"({atom_ref.element.name}) -> "
                f"#{pose_id}:{atom_pose.residue.name}{atom_pose.residue.number}@{atom_pose.name} "
                f"({atom_pose.element.name})  d={distance:.3f} A"
            )

    matched_ref = coords_ref[row_ind]
    matched_pose = coords_pose[col_ind]
    per_atom = np.linalg.norm(matched_ref - matched_pose, axis=1)
    rmsd = float(np.sqrt(np.mean(per_atom ** 2)))

    print(f"RMSD #{reference_id} vs #{pose_id} = {rmsd:.3f} A")
    print(f"  max atomic deviation : {per_atom.max():.3f} A")
    print(f"  min atomic deviation : {per_atom.min():.3f} A")
    print(f"  mean deviation       : {per_atom.mean():.3f} A")

    return rmsd, None


def calc_all_rmsd(session, reference_id=REFERENCE_MODEL_ID):
    """Compare every available pose model against one fixed reference model."""
    models = get_atomic_models(session)
    available_ids = sorted(models)

    if reference_id not in models:
        print(f"ERROR: reference model #{reference_id} was not found.")
        print(f"Available atomic models: {available_ids}")
        return {}

    pose_ids = [model_id for model_id in available_ids if model_id != reference_id]
    if not pose_ids:
        print(f"ERROR: no pose models were found to compare with reference #{reference_id}.")
        return {}

    print("\n" + "=" * 68)
    print("MULTI-POSE LIGAND RMSD")
    print(f"Fixed reference: #{reference_id} ({models[reference_id].name})")
    print("Poses to compare: " + ", ".join(f"#{model_id}" for model_id in pose_ids))
    print("=" * 68)

    results = {}
    errors = {}

    for pose_id in pose_ids:
        rmsd, error = calc_pair_rmsd(
            models[reference_id],
            models[pose_id],
            reference_id,
            pose_id,
        )
        if rmsd is not None:
            results[pose_id] = rmsd
        else:
            errors[pose_id] = error

    print("\n" + "=" * 68)
    print("FINAL RMSD SUMMARY")
    print("=" * 68)
    print(f"{'Pose':<10}{'Model name':<36}{'RMSD (A)':>12}")
    print("-" * 68)

    for pose_id in pose_ids:
        model_name = str(models[pose_id].name)
        if pose_id in results:
            print(f"#{pose_id:<9}{model_name[:35]:<36}{results[pose_id]:>12.3f}")
        else:
            print(f"#{pose_id:<9}{model_name[:35]:<36}{'ERROR':>12}")

    if results:
        best_pose_id = min(results, key=results.get)
        print("-" * 68)
        print(
            f"Lowest RMSD: #{best_pose_id} ({models[best_pose_id].name}) "
            f"= {results[best_pose_id]:.3f} A"
        )

    if errors:
        print("\nSkipped poses:")
        for pose_id in pose_ids:
            if pose_id in errors:
                print(f"  #{pose_id}: {errors[pose_id]}")

    print("=" * 68 + "\n")
    return results


# Run automatically when opened/executed inside ChimeraX.
calc_all_rmsd(session)
