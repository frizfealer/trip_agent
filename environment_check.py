#!/usr/bin/env python3
# environment_check.py - Verify environment configuration for Trip Agent

import os
import platform
import subprocess
import sys
from typing import List, Tuple

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_colored(text: str, color: str, bold: bool = False) -> None:
    """Print colored text to the terminal."""
    if bold:
        print(f"{BOLD}{color}{text}{RESET}")
    else:
        print(f"{color}{text}{RESET}")


def check_python_version() -> bool:
    """Check if Python version is 3.8+"""
    major, minor, _ = platform.python_version_tuple()
    version_ok = int(major) >= 3 and int(minor) >= 8
    if version_ok:
        print_colored(f"✓ Python version: {platform.python_version()}", GREEN)
    else:
        print_colored(f"✗ Python version: {platform.python_version()} (required: 3.8+)", RED)
    return version_ok


def check_package_installation() -> Tuple[bool, List[str]]:
    """Check if required packages are installed."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "openai",
        "python-dotenv",
        "langgraph",
        "googlemaps",
        "requests",
        "httpx",
    ]
    missing_packages = []

    print("Checking required packages:")
    for package in required_packages:
        try:
            __import__(package)
            print_colored(f"  ✓ {package}", GREEN)
        except ImportError:
            print_colored(f"  ✗ {package}", RED)
            missing_packages.append(package)

    return len(missing_packages) == 0, missing_packages


def check_environment_variables() -> Tuple[bool, List[str]]:
    """Check if required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "GOOGLE_PLACE_API_KEY",
    ]
    missing_vars = []

    print("Checking environment variables:")
    for var in required_vars:
        if os.environ.get(var):
            print_colored(f"  ✓ {var}", GREEN)
        else:
            # Check if var exists in .env file
            env_file_path = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_file_path):
                with open(env_file_path, "r") as f:
                    env_content = f.read()
                    if f"{var}=" in env_content:
                        print_colored(f"  ✓ {var} (in .env file)", YELLOW)
                        continue

            print_colored(f"  ✗ {var}", RED)
            missing_vars.append(var)

    return len(missing_vars) == 0, missing_vars


def install_missing_packages(packages: List[str]) -> bool:
    """Install missing packages."""
    if not packages:
        return True

    print_colored("\nInstalling missing packages...", YELLOW)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        return True
    except subprocess.CalledProcessError:
        print_colored("Failed to install packages. Please install them manually:", RED)
        for package in packages:
            print(f"  pip install {package}")
        return False


def main() -> None:
    """Main function."""
    print_colored("Trip Agent Environment Check", GREEN, bold=True)
    print("=" * 50)

    # Check Python version
    python_ok = check_python_version()

    # Check package installation
    print()
    packages_ok, missing_packages = check_package_installation()

    # Check environment variables
    print()
    env_vars_ok, missing_vars = check_environment_variables()

    # Summary
    print("\n" + "=" * 50)
    if python_ok and packages_ok and env_vars_ok:
        print_colored("✓ All checks passed! Your environment is ready.", GREEN, bold=True)
    else:
        print_colored("✗ Some checks failed. Please fix the issues below:", RED, bold=True)

        if not python_ok:
            print_colored("- Update Python to version 3.8 or higher", RED)

        if not packages_ok:
            print_colored(f"- Install missing packages: {', '.join(missing_packages)}", RED)
            choice = input("\nDo you want to install missing packages now? (y/n): ")
            if choice.lower() == "y":
                success = install_missing_packages(missing_packages)
                if success:
                    print_colored("✓ Packages installed successfully!", GREEN)

        if not env_vars_ok:
            print_colored("- Set required environment variables:", RED)
            for var in missing_vars:
                print(f"  export {var}=your_value")
            print("\nTip: You can add these to your .env file")


if __name__ == "__main__":
    main()
