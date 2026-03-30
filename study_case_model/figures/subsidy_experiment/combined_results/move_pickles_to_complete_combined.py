from pathlib import Path
import shutil
import re

def get_next_run_index(sub_dst_path):
    """Find next available run index in destination (run0, run1, ...)"""
    existing = []
    for d in sub_dst_path.glob("run*"):
        match = re.match(r"run(\d+)", d.name)
        if match:
            existing.append(int(match.group(1)))
    return max(existing) + 1 if existing else 0

def copy_run0_from_src1(src_path, dst_base):
    """Copy run0 from src1, replacing existing run0 in destination"""
    for dev_dir in src_path.glob("dev*/"):
        for sub_dir in dev_dir.glob("sub*/"):
            src_run0 = sub_dir / "run0"
            if not src_run0.exists():
                continue

            dst_run0 = dst_base / dev_dir.name / sub_dir.name / "run0"
            dst_run0.parent.mkdir(parents=True, exist_ok=True)

            if dst_run0.exists():
                shutil.rmtree(dst_run0)

            shutil.copytree(src_run0, dst_run0)
            print(f"[SRC1] {src_run0} → {dst_run0}")

def merge_runs(src_paths, dst_base):
    for src in src_paths:
        print(f"\nProcessing source: {src}")

        for dev_dir in src.glob("dev*/"):
            for sub_dir in dev_dir.glob("sub*/"):

                dst_sub = dst_base / dev_dir.name / sub_dir.name
                dst_sub.mkdir(parents=True, exist_ok=True)

                # find next available index
                next_idx = get_next_run_index(dst_sub)

                for run_dir in sorted(sub_dir.glob("run*/")):
                    dst_run = dst_sub / f"run{next_idx}"

                    shutil.copytree(run_dir, dst_run)
                    print(f"{run_dir} → {dst_run}")

                    next_idx += 1


# ---------------------------
# Paths
# ---------------------------
src1 = Path("study_case_model/figures/subsidy_experiment/run_24326")
src2 = Path("study_case_model/figures/subsidy_experiment/run_27326")
src3 = Path("study_case_model/figures/subsidy_experiment/run_29326")
dst_base = Path("study_case_model/figures/subsidy_experiment/combined_runs")

# ---------------------------
# Run merge
# ---------------------------
src_run_zero = Path("study_case_model/figures/subsidy_experiment/run_24326_0")
copy_run0_from_src1(src1, src_run_zero)
merge_runs([src_run_zero, src2, src3], dst_base)