import subprocess
import sys

def run_script(script_name):
    print(f"\nRunning {script_name}...\n")

    result = subprocess.run(
        [sys.executable, script_name],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    if result.returncode != 0:
        print(f"\nError: {script_name} failed with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nFinished {script_name}\n")


if __name__ == "__main__":
    run_script("flux.py")
    run_script("regen.py")

    print("Pipeline completed successfully.")
