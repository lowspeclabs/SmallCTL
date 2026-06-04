#!/usr/bin/env python3
"""Verify Phase 2: run behavioral tests."""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Run Phase 2 behavioral tests
result = os.system(f'cd {script_dir} && python test_phase2.py')

print("=== Phase 2 Test Output ===")
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print(f"\nExit code: {result.returncode}")

if result.returncode == 0:
    print("\n✓ PHASE 2 IS COMPLETE - All tests passed")
else:
    print("\n✗ PHASE 2 NOT COMPLETE - Tests failed")
