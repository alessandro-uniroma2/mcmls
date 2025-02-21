import os
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import zipfile

from factory.models import ModelFactory
from mcmls import McMls
from utils.common import get_project_root
from utils.downloader import Downloader


def extract_zip(archive_file, output_dir):
    """
    Extracts the contents of a zip archive to a specified output directory.

    Args:
        archive_file (Path): Path to the zip archive file.
        output_dir (Path): Path to the output directory where the files should be extracted.
    """
    if not zipfile.is_zipfile(archive_file):
        print(f"{archive_file} is not a valid zip file.")
        return

    try:
        with zipfile.ZipFile(archive_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            print(f"Extracted {archive_file} to {output_dir}")
    except zipfile.BadZipFile:
        print(f"Error: {archive_file} is corrupted or not a zip file.")
    except Exception as e:
        print(f"An error occurred while extracting {archive_file}: {e}")
    print('')


def extract_code_from_ipynb(ipynb_file, output_file):
    try:
        import nbformat
        with open(ipynb_file, 'r', encoding='utf-8') as file:
            notebook = nbformat.read(file, as_version=4)

        code_cells = [cell['source'] for cell in notebook['cells'] if cell['cell_type'] == 'code']

        with open(output_file, 'w', encoding='utf-8') as file:
            for i, code in enumerate(code_cells, 1):
                file.write(code)
                file.write('\n\n')
    except ImportError:
        print("[ERROR] Package nbformat missing. Please install the environment first")
        exit(1)


def sha1sum(filename):
    """Returns the SHA-1 hash of a file."""
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while chunk := f.read(8192):
            sha1.update(chunk)
    return sha1.hexdigest()


def fetch_datasets(config_file_path):
    """Fetch datasets if not already present."""
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    data_dir = get_project_root().joinpath("mcmls", "archive")
    data_dir.mkdir(exist_ok=True)

    for dataset in config.get('datasets', []):
        url = dataset.get("url")
        dataset_path = Path(dataset.get("archive"))
        name = dataset.get("name")
        if not dataset_path.exists():
            print(f"Downloading {name} from {url}...")
            downloader = Downloader(download_directory=str(data_dir))
            downloader.download(url)
        else:
            print(f"{name} already exists.")

    check_archives(config_file_path)


def check_archives(config_file_path):
    """Check and print dataset details."""
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    data_dir = Path("data")

    # Preliminary Check
    errors = 0
    for dataset in config.get('datasets', []):
        archive_file = Path(dataset.get("archive"))
        if not archive_file.is_file():
            errors += 1

    if errors > 0:
        return fetch_datasets(config_file_path)

    for dataset in config.get('datasets', []):
        archive_file = Path(dataset.get("archive"))
        dataset_name = dataset.get("name")
        dataset_path = data_dir.joinpath(dataset_name + "_dataset")

        if dataset_path.exists():
            last_modified = datetime.fromtimestamp(archive_file.stat().st_mtime)
            print(f"Dataset: {dataset_name}")
            print(f"Last Modified: {last_modified}\n")

        elif not dataset_path.is_dir() and archive_file.is_file():
            size = archive_file.stat().st_size
            sha1_hash = sha1sum(archive_file)
            last_modified = datetime.fromtimestamp(archive_file.stat().st_mtime)
            print(f"Archive: {dataset_name}")
            print(f"Size: {size} bytes")
            print(f"SHA-1: {sha1_hash}")
            print(f"Last Modified: {last_modified}\n")
            if sha1_hash == dataset.get("hash"):
                print("--- Archive validated. Extracting...")
                extract_zip(archive_file, data_dir)
                print("--- Done")

        else:
            print(f"{dataset_name} not found. Download manually at {dataset.get('url')}")


def extract_notebook(notebook_name):
    """Extracts code from notebook to a .py file."""
    notebook_path = Path("notebooks") / notebook_name
    if not notebook_path.exists():
        print(f"Notebook {notebook_name} does not exist.")
        return

    output_path = notebook_path.with_suffix('.py')
    extract_code_from_ipynb(notebook_path, output_path)
    print(f"Code extracted to {output_path}")


def generate_notebook_from_files(notebook_name):
    """Extracts code from the project files to notebook."""
    project_root = get_project_root().joinpath("mcmls")

    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(exist_ok=True)
    notebook_path = notebook_dir.joinpath(notebook_name)

    all_files = project_root.rglob("*.py")
    codes = {}
    imports = []

    for file in all_files:
        try:
            codes[str(file)] = file.read_text()
        except:
            continue
    with open(notebook_path, "w") as f:
        json.dump(codes, f)

    print(json.dumps(codes, indent=2))


def check_env_requirements():
    """Checks if all requirements in requirements.txt are installed."""
    with open("requirements.txt", 'r') as f:
        requirements = f.read().splitlines()
    errors = 0
    for req in requirements:
        try:
            subprocess.check_call([f"pip", "show", req], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            errors += 1
            print(f"[-] Requirement not satisfied: {req}")

    if errors == 0:
        print("[+] All requirements satisfied.")

def install_requirements():
    """Install all requirements from requirements.txt."""
    subprocess.run(["pip", "install", "-r", "requirements.txt"])


def parse_args() -> argparse.Namespace:
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Malware Classifier Project Utility Script")
    subparsers = parser.add_subparsers(dest="action", help="Actions to perform")

    # Dataset action
    dataset_parser = subparsers.add_parser('dataset', help="Dataset related actions")
    dataset_subparsers = dataset_parser.add_subparsers(dest="command", required=True)

    # dataset fetch
    fetch_parser = dataset_subparsers.add_parser('fetch', help="Download datasets")
    fetch_parser.add_argument('--config', type=str, default='config/dataset_config.json', help="Path to config file")

    # dataset check
    check_parser = dataset_subparsers.add_parser('check', help="Check datasets")
    check_parser.add_argument('--config', type=str, default='config/dataset_config.json', help="Path to config file")

    # Notebook action
    notebook_parser = subparsers.add_parser('notebook', help="Notebook related actions")
    notebook_subparsers = notebook_parser.add_subparsers(dest="command", required=True)

    # notebook extract
    extract_parser = notebook_subparsers.add_parser('extract', help="Extract code from notebook")
    extract_parser.add_argument('notebook', type=str, help="Notebook filename (with .ipynb extension)")

    # notebook generate
    generate_parser = notebook_subparsers.add_parser('generate', help="Generate notebook from code")
    generate_parser.add_argument('notebook', type=str, help="Notebook filename (with .ipynb extension)")

    # Environment action
    env_parser = subparsers.add_parser('env', help="Environment related actions")
    env_subparsers = env_parser.add_subparsers(dest="command", required=True)

    # env check
    env_subparsers.add_parser('check', help="Check if requirements are installed")

    # env install
    env_subparsers.add_parser('install', help="Install requirements from requirements.txt")

    model_parser = subparsers.add_parser('model', help="Model related actions")
    model_subparsers = model_parser.add_subparsers(dest="command", required=True)

    model_train_parser = model_subparsers.add_parser('train', help="Train model with specific dataset or on all datasets")
    model_train_parser.add_argument("-m", "--model", nargs='+', action='extend', choices=ModelFactory.list_all() + [None], help="Model to run", default=None)
    model_train_parser.add_argument("-d", "--dataset", nargs='+', action='extend', help="Dataset to train with", default=None)

    model_run_parser = model_subparsers.add_parser('run', help="Run predictions and show benchmarks")
    model_run_parser.add_argument("-m", "--model", nargs='+', action='extend', help="Model to run", choices=ModelFactory.list_all() + [None], default=None)
    model_run_parser.add_argument("-d", "--dataset", nargs='+', action='extend', help="Dataset to run with", default=None)

    model_compare_parser = model_subparsers.add_parser('compare', help="Show benchmarks")
    model_compare_parser.add_argument("-m", "--model", nargs='+', action='extend', help="Model to compare", choices=ModelFactory.list_all() + [None], default=None)
    model_compare_parser.add_argument("-d", "--dataset", nargs='+', action='extend', help="Dataset to compare on", default=None)

    model_subparsers.add_parser('list', help="List available models")

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    
    if args.action == "dataset":
        if args.command == "fetch":
            fetch_datasets(args.config)
        elif args.command == "check":
            check_archives(args.config)

    elif args.action == "notebook":
        if args.command == "extract":
            extract_notebook(args.notebook)

        elif args.command == "generate":
            generate_notebook_from_files(args.notebook)

    elif args.action == "env":
        if args.command == "check":
            check_env_requirements()
        elif args.command == "install":
            install_requirements()

    elif args.action == "model":

        if args.command in ["train", "run", "compare"]:
            mcmls = McMls(results_dir="results")
            model_under_test = ModelFactory.list_all()
            if args.model is not None:
                model_under_test = [args.model]
            dataset_under_test = mcmls.loader.list_datasets()
            if args.dataset is not None:
                dataset_under_test = [args.dataset]

            if args.command == "run":
                # Load testing data
                mcmls.evaluate_subset(model_names=model_under_test, dataset_names=dataset_under_test)
                mcmls.save_results()
            elif args.command == "train":
                mcmls.evaluate_subset(model_names=model_under_test, dataset_names=dataset_under_test)
                mcmls.save_results()
            elif args.command == "compare":
                mcmls.load_results()
                mcmls.summarize_results()

                for model in mcmls.list_models():
                    print("Confusion Matrix for Model:", model)
                    mcmls.plot_confusion_matrices(model)

    elif args.action == "list":
            print("(*) Available models:")
            for model in ModelFactory.list_all():
                print(" - " + model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

