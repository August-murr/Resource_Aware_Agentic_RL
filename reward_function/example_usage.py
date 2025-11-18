"""
Example usage of the Resource-Aware Solution Evaluator

This script shows how to evaluate a single solution and get its reward.
"""

import os
from datasets import load_dataset
from reward_function.solution_evaluator import ResourceAwareEvaluator

# Set your E2B API key
os.environ["E2B_API_KEY"] = "YOUR_KEY_HERE"

# Load dataset
print("Loading dataset...")
dataset = load_dataset("newfacade/LeetCodeDataset", split="train")
problem = dataset[0]  # Get first problem (Two Sum)

print(f"Problem: {problem['task_id']}\n")

# Your solution code
my_solution = '''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, x in enumerate(nums):
            if (y := target - x) in d:
                return [d[y], i]
            d[x] = i
'''

# Evaluate the solution
print("Evaluating solution...")
with ResourceAwareEvaluator(timeout=120) as evaluator:
    result = evaluator.evaluate_solution(
        problem=problem,
        solution_code=my_solution,
        num_runs=1  # Use num_runs=3-5 for more stable metrics
    )

# Print results
print(result)

# Access individual metrics
print(f"\nReward: {result.reward}")
print(f"All tests passed: {result.passed}")
print(f"Tests passed: {result.num_passed}/{result.total_tests}")

if result.execution_time:
    print(f"Execution time: {result.execution_time:.6f} seconds")
if result.peak_memory:
    print(f"Peak memory: {result.peak_memory:,} bytes")
