#!/usr/bin/env python3
"""
Test runner script for ngram-predictor project.

This script runs all unit tests and provides coverage reporting.
Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --coverage   # Run tests with coverage report
    python run_tests.py --fast       # Run only fast tests (exclude slow ones)
"""

import sys
import subprocess
import argparse


def run_tests(coverage=False, fast=False):
    """Run the test suite with optional coverage reporting."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add coverage options if requested
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-branch"
        ])
    
    # Exclude slow tests if fast mode is requested
    if fast:
        cmd.extend(["-m", "not slow"])
    
    # Add the tests directory
    cmd.append("tests/")
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Tests failed with return code: {e.returncode}")
        return e.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install it with: pip install pytest")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run tests for ngram-predictor")
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Generate coverage report (requires pytest-cov)"
    )
    parser.add_argument(
        "--fast", 
        action="store_true", 
        help="Run only fast tests (exclude slow tests)"
    )
    
    args = parser.parse_args()
    
    # Install coverage dependencies if coverage is requested
    if args.coverage:
        try:
            import pytest_cov
        except ImportError:
            print("Coverage reporting requires pytest-cov. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pytest-cov"], check=True)
    
    return run_tests(coverage=args.coverage, fast=args.fast)


if __name__ == "__main__":
    sys.exit(main()) 