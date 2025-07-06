import shutil
from pathlib import Path
import sys

'''
Based on which fold 1-5 as test is given, moves fold files in directories train/ and val/ where the dataloader works on.
'''

def prepare_folds(fold_val: int, base_dir: str = "data/dental"):
    assert 1 <= fold_val <= 5, "fold_val not 1-5"
    base_path = Path(base_dir)

    train_dir = base_path / "train"
    val_dir = base_path / "val"

    for d in [train_dir, val_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for i in range(1, 6):
        fold_dir = base_path / f"fold_{i}"
        assert fold_dir.exists(), f"{fold_dir} not found"

        target_dir = val_dir if i == fold_val else train_dir
        for file in fold_dir.glob("*.json"):
            shutil.copy(file, target_dir / file.name)

    print(f"Fold {fold_val} is set as val.")

if __name__ == "__main__":
    prepare_folds(int(sys.argv[1]), sys.argv[2])