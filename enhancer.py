import subprocess
import sys
import yaml

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
    with open("options.yaml", "r") as f:
        config = yaml.safe_load(f)

    im2im_model = config["im2im_model"]

    run_script("flux.py")
    
    print(im2im_model)
    if im2im_model == "regen":
        run_script("regen.py")
    elif im2im_model == "hypergan":
        run_script("hypergan.py")
    else:
        print("The available im2im translation models are ['regen','hypergan'].")

    print("Pipeline completed successfully.")
