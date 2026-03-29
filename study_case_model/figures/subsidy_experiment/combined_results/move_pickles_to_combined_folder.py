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


src1 = Path("study_case_model/figures/subsidy_experiment/run_24326")
src2 = Path("study_case_model/figures/subsidy_experiment/run_27326")
dst_base = Path("study_case_model/figures/subsidy_experiment/combined_runs")

# ---------------------------
# PART 1: Copy run0 → run2
# ---------------------------
for dev_dir in src1.glob("dev*/"):
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


# ---------------------------
# PART 2: Append all runs from second source
# ---------------------------
for dev_dir in src2.glob("dev*/"):
    for sub_dir in dev_dir.glob("sub*/"):

        dst_sub = dst_base / dev_dir.name / sub_dir.name
        dst_sub.mkdir(parents=True, exist_ok=True)

        # get next available run index in destination
        next_idx = get_next_run_index(dst_sub)

        for run_dir in sorted(sub_dir.glob("run*/")):
            dst_run = dst_sub / f"run{next_idx}"

            shutil.copytree(run_dir, dst_run)
            print(f"[SRC2] {run_dir} → {dst_run}")

            next_idx += 1