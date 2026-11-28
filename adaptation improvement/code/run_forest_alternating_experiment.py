"""Launch four alternating forest variants on the existing local splits."""
from run_pairwise_redesign_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(driver="tools/t1_forest_alternating.py"))
