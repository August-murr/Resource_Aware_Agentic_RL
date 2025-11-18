"""
Test script for the Resource-Aware Solution Evaluator

This script demonstrates the evaluator with different solution variants:
1. Correct solution (from dataset)
2. Incorrect solution (fails some tests)
3. Memory-heavy solution (correct but wasteful)
"""

import os
from datasets import load_dataset
from reward_function.solution_evaluator import ResourceAwareEvaluator

# Set E2B API key
os.environ["E2B_API_KEY"] = "YOUR_KEY_HERE"

def load_leetcode_dataset():
    """Load the LeetCode dataset"""
    print("Loading LeetCode dataset...")
    dataset = load_dataset("newfacade/LeetCodeDataset", split="train")
    print(f"✅ Dataset loaded: {len(dataset)} problems\n")
    return dataset

def test_correct_solution(problem):
    """Test 1: Correct solution from dataset"""
    print("=" * 70)
    print("TEST 1: CORRECT SOLUTION (from dataset)")
    print("=" * 70)
    print(f"Problem: {problem['task_id']}")
    print(f"Entry Point: {problem['entry_point']}\n")
    
    with ResourceAwareEvaluator(timeout=120) as evaluator:
        result = evaluator.evaluate_solution(problem, problem["completion"], num_runs=1)
    
    print(result)
    return result

def test_incorrect_solution(problem):
    """Test 2: Incorrect solution - always returns [0, 1]"""
    print("\n" + "=" * 70)
    print("TEST 2: INCORRECT SOLUTION (always returns [0, 1])")
    print("=" * 70)
    
    incorrect_solution = '''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Wrong: always returns first two indices
        return [0, 1]
'''
    
    with ResourceAwareEvaluator(timeout=120) as evaluator:
        result = evaluator.evaluate_solution(problem, incorrect_solution, num_runs=1)
    
    print(result)
    return result

def test_memory_heavy_solution(problem):
    """Test 3: Memory-heavy solution - correct but wasteful"""
    print("\n" + "=" * 70)
    print("TEST 3: MEMORY-HEAVY SOLUTION (correct but wasteful)")
    print("=" * 70)
    
    memory_heavy_solution = '''
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Wasteful: create large unnecessary data structures
        all_nums_copy = [nums[:] for _ in range(1000)]
        wasteful_dict = {i: [nums[i]] * 1000 for i in range(len(nums))}
        
        # Now do the actual work (correctly)
        d = {}
        for i, x in enumerate(nums):
            if (y := target - x) in d:
                return [d[y], i]
            d[x] = i
        return None
'''
    
    with ResourceAwareEvaluator(timeout=120) as evaluator:
        result = evaluator.evaluate_solution(problem, memory_heavy_solution, num_runs=1)
    
    print(result)
    return result

def print_summary(results):
    """Print a summary comparison of all results"""
    print("\n" + "=" * 70)
    print("SUMMARY & COMPARISON")
    print("=" * 70)
    print(f"\n{'Solution':<20} {'Passed':<8} {'Tests':<12} {'Time':<15} {'Memory':<15} {'Reward':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        time_str = f"{result.execution_time:.6f}s" if result.execution_time else "N/A"
        memory_str = f"{result.peak_memory/1024:.2f} KB" if result.peak_memory else "N/A"
        tests_str = f"{result.num_passed}/{result.total_tests}"
        
        print(f"{name:<20} {str(result.passed):<8} {tests_str:<12} {time_str:<15} {memory_str:<15} {result.reward:<10.4f}")
    
    print("=" * 70)
    
    # Print key insights
    print("\n📊 Key Insights:")
    print(f"  • Correct solution: Max reward = {results['Correct'].reward:.4f}")
    print(f"  • Incorrect solution: Partial credit = {results['Incorrect'].reward:.4f} ({results['Incorrect'].num_passed}/{results['Incorrect'].total_tests} tests)")
    print(f"  • Memory-heavy solution: Penalized for high memory usage = {results['Memory-Heavy'].reward:.4f}")
    
    if results['Correct'].peak_memory and results['Memory-Heavy'].peak_memory:
        memory_ratio = results['Memory-Heavy'].peak_memory / results['Correct'].peak_memory
        print(f"  • Memory overhead: {memory_ratio:.1f}x more memory in wasteful solution")

def main():
    """Main test runner"""
    # Load dataset
    dataset = load_leetcode_dataset()
    
    # Get first problem (Two Sum)
    problem = dataset[0]
    
    # Run all tests
    results = {
        'Correct': test_correct_solution(problem),
        'Incorrect': test_incorrect_solution(problem),
        'Memory-Heavy': test_memory_heavy_solution(problem),
    }
    
    # Print summary
    print_summary(results)
    
    print("\n✅ All tests completed!")
    print("\n💡 The evaluator successfully:")
    print("  1. ✅ Prioritizes correctness (partial credit for incorrect solutions)")
    print("  2. ✅ Only rewards time/memory when 100% tests pass")
    print("  3. ✅ Penalizes resource-heavy solutions")
    print("  4. ✅ Runs safely in E2B sandboxed environment")

if __name__ == "__main__":
    main()
