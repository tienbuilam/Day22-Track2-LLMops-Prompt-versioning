import argparse
import subprocess
import sys

def run_step(script_name):
    print(f"\n{'='*60}")
    print(f"▶️ Running {script_name}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"❌ Error running {script_name}. Exiting.")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=int, help="Run a specific step (1-4)")
    args = parser.parse_args()

    steps = [
        "01_langsmith_rag_pipeline.py",
        "02_prompt_hub_ab_routing.py",
        "03_ragas_evaluation.py",
        "04_guardrails_validator.py"
    ]

    if args.step:
        if 1 <= args.step <= 4:
            run_step(steps[args.step - 1])
        else:
            print("Invalid step. Choose between 1 and 4.")
    else:
        for step in steps:
            run_step(step)

    print("\n✅ All requested steps completed!")

if __name__ == "__main__":
    main()
