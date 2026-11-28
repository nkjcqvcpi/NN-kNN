"""Launch the matched Random Forest pair experiment on local CPU."""
from run_pairwise_redesign_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(driver="tools/t1_pair_random_forest.py"))
