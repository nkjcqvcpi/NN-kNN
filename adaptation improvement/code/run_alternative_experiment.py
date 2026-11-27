"""Launch the recorded follow-up alternatives using the shared local CPU runner."""
from run_pairwise_redesign_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(driver="tools/t1_alternative_redesign.py"))
