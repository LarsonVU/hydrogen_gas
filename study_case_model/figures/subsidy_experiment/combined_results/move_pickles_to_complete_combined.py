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

def copy_run_from_src1(src_path, dst_base, run = "run0"):
    """Copy run0 from src1, replacing existing run0 in destination"""
    for dev_dir in src_path.glob("dev*/"):
        for sub_dir in dev_dir.glob("sub*/"):
            src_run0 = sub_dir / run
            if not src_run0.exists():
                continue

            dst_run0 = dst_base / dev_dir.name / sub_dir.name / run
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
src1 = Path("study_case_model/figures/subsidy_experiment/run_06426")
dst_base = Path("study_case_model/figures/subsidy_experiment/combined_runs_new")

# ---------------------------
# Run merge
# ---------------------------
src_run_five = Path("study_case_model/figures/subsidy_experiment/run_5")
src_run_six = Path("study_case_model/figures/subsidy_experiment/run_6")
src_run_seven = Path("study_case_model/figures/subsidy_experiment/run_7")
copy_run_from_src1(src1, src_run_five, run="run5")
copy_run_from_src1(src1, src_run_six, run="run6")
copy_run_from_src1(src1, src_run_seven, run="run7")

merge_runs([src_run_five, src_run_six, src_run_seven], dst_base)