"""
Resource-Aware Solution Evaluator for LeetCode Problems

This module provides a evaluator that:
- Runs code in E2B sandboxed environment
- Measures execution time and peak memory usage
- Prioritizes correctness with partial credit
- Only rewards time/memory efficiency when all tests pass
"""

import os
import time
import statistics
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from e2b_code_interpreter import Sandbox


@dataclass
class EvaluationResult:
    """Results from evaluating a solution"""
    passed: bool
    num_passed: int
    total_tests: int
    execution_time: Optional[float] = None  # Only if all tests passed
    peak_memory: Optional[int] = None  # Only if all tests passed (in bytes)
    reward: float = 0.0
    error_message: Optional[str] = None

    def __str__(self):
        """Pretty print the results"""
        lines = [
            "=" * 60,
            f"✅ Passed: {self.passed}",
            f"📊 Tests: {self.num_passed}/{self.total_tests}",
        ]
        
        if self.execution_time is not None:
            lines.append(f"⏱️  Time: {self.execution_time:.6f}s")
        else:
            lines.append("⏱️  Time: N/A")
            
        if self.peak_memory is not None:
            lines.append(f"💾 Memory: {self.peak_memory:,} bytes ({self.peak_memory/1024:.2f} KB)")
        else:
            lines.append("💾 Memory: N/A")
            
        lines.append(f"🎁 Reward: {self.reward:.4f}")
        
        if self.error_message:
            lines.append(f"❌ Error: {self.error_message[:200]}")
            
        lines.append("=" * 60)
        return "\n".join(lines)


class ResourceAwareEvaluator:
    """
    Evaluates LeetCode solutions with resource awareness.
    Prioritizes correctness, only rewards time/memory if 100% correct.
    
    Usage:
        with ResourceAwareEvaluator(timeout=120) as evaluator:
            result = evaluator.evaluate_solution(problem, solution_code, num_runs=1)
            print(f"Reward: {result.reward}")
    """
    
    def __init__(self, timeout: int = 120):
        """
        Args:
            timeout: Maximum execution time in seconds (default 2 minutes)
        """
        self.timeout = timeout
        self.sandbox = None
    
    def __enter__(self):
        """Context manager entry - create sandbox"""
        self.sandbox = Sandbox.create()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup sandbox"""
        if self.sandbox:
            self.sandbox.kill()
    
    def create_evaluation_code(self, problem: Dict, solution_code: str) -> str:
        """
        Constructs Python code that runs in E2B sandbox with metrics tracking.
        
        Args:
            problem: Problem dictionary with 'prompt', 'test', and 'entry_point'
            solution_code: The solution code to evaluate
            
        Returns:
            Complete Python code to execute in sandbox
        """
        test_func = problem["test"]
        entry_point = problem["entry_point"]
        
        code = f'''
import time
import tracemalloc
import sys

{problem["prompt"]}

{solution_code}

# Extract test cases
test_function_code = """{test_func}"""

# Parse individual assert statements
import re
assert_pattern = r'assert\\s+candidate\\([^)]+\\)[^\\n]+'
test_cases = re.findall(assert_pattern, test_function_code)

# Results tracking
results = {{
    "total": len(test_cases),
    "passed": 0,
    "failed": 0,
    "times": [],
    "memories": [],
    "errors": []
}}

# Run each test case individually
for i, test_case in enumerate(test_cases):
    try:
        # Start tracking
        tracemalloc.start()
        start_time = time.perf_counter()
        
        # Execute the test
        candidate = {entry_point}
        exec(test_case)
        
        # Stop tracking
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Record metrics
        results["passed"] += 1
        results["times"].append(end_time - start_time)
        results["memories"].append(peak)
        
    except AssertionError as e:
        tracemalloc.stop() if tracemalloc.is_tracing() else None
        results["failed"] += 1
        results["errors"].append(f"Test {{i+1}}: Assertion failed")
    except Exception as e:
        tracemalloc.stop() if tracemalloc.is_tracing() else None
        results["failed"] += 1
        results["errors"].append(f"Test {{i+1}}: {{type(e).__name__}}: {{str(e)}}")

# Output results in parseable format
print("RESULTS_START")
print(f"TOTAL:{{results['total']}}")
print(f"PASSED:{{results['passed']}}")
print(f"FAILED:{{results['failed']}}")
if results["passed"] == results["total"]:
    avg_time = sum(results["times"]) / len(results["times"]) if results["times"] else 0
    max_memory = max(results["memories"]) if results["memories"] else 0
    print(f"TIME:{{avg_time}}")
    print(f"MEMORY:{{max_memory}}")
if results["errors"]:
    print("ERRORS:" + ";".join(results["errors"]))
print("RESULTS_END")
'''
        return code
    
    def parse_output(self, output: str) -> Dict:
        """
        Parse the output from E2B execution
        
        Args:
            output: Combined stdout and stderr from execution
            
        Returns:
            Dictionary with parsed metrics
        """
        results = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "time": None,
            "memory": None,
            "errors": []
        }
        
        # Extract results between markers
        match = re.search(r'RESULTS_START(.+?)RESULTS_END', output, re.DOTALL)
        if not match:
            return results
        
        result_text = match.group(1)
        
        # Parse each metric
        for line in result_text.strip().split('\n'):
            if line.startswith('TOTAL:'):
                results["total"] = int(line.split(':')[1])
            elif line.startswith('PASSED:'):
                results["passed"] = int(line.split(':')[1])
            elif line.startswith('FAILED:'):
                results["failed"] = int(line.split(':')[1])
            elif line.startswith('TIME:'):
                results["time"] = float(line.split(':')[1])
            elif line.startswith('MEMORY:'):
                results["memory"] = int(line.split(':')[1])
            elif line.startswith('ERRORS:'):
                errors_str = line.split(':', 1)[1]
                results["errors"] = errors_str.split(';') if errors_str else []
        
        return results
    
    def calculate_reward(self, total: int, passed: int, exec_time: Optional[float], 
                        memory: Optional[int]) -> float:
        """
        Calculate reward based on correctness-first strategy.
        
        Reward Structure:
        - If all tests pass: reward = 1.0 + time_bonus + memory_bonus (unbounded)
        - If partial: reward = fraction of tests passed (no time/memory bonus)
        
        Time bonus uses inverse relationship: baseline_time / exec_time
        Memory bonus uses inverse relationship: baseline_memory / memory
        
        Args:
            total: Total number of test cases
            passed: Number of test cases passed
            exec_time: Average execution time (only if all passed)
            memory: Peak memory usage in bytes (only if all passed)
            
        Returns:
            Reward value - unbounded for perfect solutions, [0.0, 1.0) for partial
        """
        if passed == total and passed > 0:
            # All tests passed - add time and memory bonuses (unbounded)
            base_reward = 1.0
            
            # Time bonus: faster is better (unbounded, inverse relationship)
            if exec_time and exec_time > 0:
                baseline_time = 0.01  # 10ms baseline
                time_bonus = baseline_time / exec_time
            else:
                time_bonus = 1.0
            
            # Memory bonus: less memory is better (unbounded, inverse relationship)
            if memory and memory > 0:
                baseline_memory = 1048576  # 1MB baseline
                memory_bonus = baseline_memory / memory
            else:
                memory_bonus = 1.0
            
            return base_reward + time_bonus + memory_bonus
        else:
            # Partial credit - only based on correctness fraction
            return passed / total if total > 0 else 0.0
    
    def evaluate_solution(self, problem: Dict, solution_code: str, 
                         num_runs: int = 1) -> EvaluationResult:
        """
        Evaluate a solution against test cases.
        
        Args:
            problem: Problem dict from dataset with 'prompt', 'test', 'entry_point'
            solution_code: The solution code to evaluate
            num_runs: Number of times to run for variance reduction (default 1)
            
        Returns:
            EvaluationResult with metrics and reward
        """
        if not self.sandbox:
            raise RuntimeError("Evaluator must be used as context manager: "
                             "with ResourceAwareEvaluator() as evaluator:")
        
        all_times = []
        all_memories = []
        
        for run in range(num_runs):
            try:
                # Create evaluation code
                eval_code = self.create_evaluation_code(problem, solution_code)
                
                # Execute in sandbox with timeout
                execution = self.sandbox.run_code(
                    eval_code,
                    timeout=self.timeout
                )
                
                # Get output (stdout/stderr can be lists)
                stdout = ''.join(execution.logs.stdout) if isinstance(execution.logs.stdout, list) else execution.logs.stdout
                stderr = ''.join(execution.logs.stderr) if isinstance(execution.logs.stderr, list) else execution.logs.stderr
                output = stdout + stderr
                
                # Check for errors
                if execution.error:
                    return EvaluationResult(
                        passed=False,
                        num_passed=0,
                        total_tests=0,
                        error_message=f"Execution error: {execution.error.name} - {execution.error.value}",
                        reward=0.0
                    )
                
                # Parse results
                results = self.parse_output(output)
                
                # If not all passed, return immediately (no point in multiple runs)
                if results["passed"] < results["total"]:
                    reward = self.calculate_reward(
                        results["total"], results["passed"], None, None
                    )
                    error_msg = "; ".join(results["errors"]) if results["errors"] else None
                    return EvaluationResult(
                        passed=False,
                        num_passed=results["passed"],
                        total_tests=results["total"],
                        reward=reward,
                        error_message=error_msg
                    )
                
                # All passed - collect metrics
                if results["time"] is not None:
                    all_times.append(results["time"])
                if results["memory"] is not None:
                    all_memories.append(results["memory"])
                    
            except Exception as e:
                return EvaluationResult(
                    passed=False,
                    num_passed=0,
                    total_tests=0,
                    error_message=f"Exception during evaluation: {str(e)}",
                    reward=0.0
                )
        
        # All runs passed - aggregate metrics
        if all_times and all_memories:
            # Use median time and max memory
            median_time = statistics.median(all_times) if len(all_times) > 1 else all_times[0]
            max_memory = max(all_memories)
            
            reward = self.calculate_reward(results["total"], results["passed"], 
                                          median_time, max_memory)
            
            return EvaluationResult(
                passed=True,
                num_passed=results["passed"],
                total_tests=results["total"],
                execution_time=median_time,
                peak_memory=max_memory,
                reward=reward
            )
        else:
            # Shouldn't happen, but handle gracefully
            return EvaluationResult(
                passed=True,
                num_passed=results["passed"],
                total_tests=results["total"],
                reward=1.0
            )
