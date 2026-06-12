"""Verifiable code-generation tasks (HumanEval/MBPP-style, RLVR-ready)."""

from __future__ import annotations

# Each task: prompt shown to model, entry_point function name, test harness with `check`.
_CODE_SEEDS: list[dict] = [
    {
        "task_id": "he_has_close_elements",
        "category": "logic",
        "difficulty": "easy",
        "prompt": (
            "Write a Python function `has_close_elements(numbers, threshold)` that returns True "
            "if any two numbers in the list are closer than threshold."
        ),
        "entry_point": "has_close_elements",
        "test": """
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) is False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) is True
""",
    },
    {
        "task_id": "he_separate_paren_groups",
        "category": "strings",
        "difficulty": "medium",
        "prompt": (
            "Write `separate_paren_groups(paren_string)` returning a list of balanced parenthesis "
            "groups separated by spaces in the input string."
        ),
        "entry_point": "separate_paren_groups",
        "test": """
def check(candidate):
    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']
""",
    },
    {
        "task_id": "he_below_zero",
        "category": "simulation",
        "difficulty": "easy",
        "prompt": (
            "Write `below_zero(operations)` where operations is a list of deposits (+) and "
            "withdrawals (-). Return True if balance ever goes below zero."
        ),
        "entry_point": "below_zero",
        "test": """
def check(candidate):
    assert candidate([1, 2, -4, 5]) is True
    assert candidate([1, 2, -3, 1, 2, -3]) is False
""",
    },
    {
        "task_id": "he_mean_absolute_deviation",
        "category": "math",
        "difficulty": "easy",
        "prompt": (
            "Write `mean_absolute_deviation(numbers)` returning the mean absolute deviation "
            "from the mean of the list."
        ),
        "entry_point": "mean_absolute_deviation",
        "test": """
def check(candidate):
    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6
""",
    },
    {
        "task_id": "he_intersection",
        "category": "intervals",
        "difficulty": "medium",
        "prompt": (
            "Write `interval_intersection(interval1, interval2)` for closed intervals [start, end]. "
            "Return None if disjoint, else the intersection interval."
        ),
        "entry_point": "interval_intersection",
        "test": """
def check(candidate):
    assert candidate((1, 2), (2, 3)) == (2, 2)
    assert candidate((1, 2), (3, 4)) is None
""",
    },
    {
        "task_id": "he_palindrome",
        "category": "strings",
        "difficulty": "easy",
        "prompt": "Write `is_palindrome(text)` returning whether the string is a palindrome (ignore case).",
        "entry_point": "is_palindrome",
        "test": """
def check(candidate):
    assert candidate('') is True
    assert candidate('aba') is True
    assert candidate('abc') is False
    assert candidate('AbA') is True
""",
    },
    {
        "task_id": "he_prime",
        "category": "math",
        "difficulty": "easy",
        "prompt": "Write `is_prime(n)` returning whether integer n >= 0 is prime.",
        "entry_point": "is_prime",
        "test": """
def check(candidate):
    assert candidate(2) is True
    assert candidate(4) is False
    assert candidate(17) is True
    assert candidate(1) is False
""",
    },
    {
        "task_id": "he_fizzbuzz_value",
        "category": "simulation",
        "difficulty": "easy",
        "prompt": (
            "Write `fizzbuzz_value(n)` returning 'fizz' if n%3==0, 'buzz' if n%5==0, "
            "'fizzbuzz' if both, else str(n)."
        ),
        "entry_point": "fizzbuzz_value",
        "test": """
def check(candidate):
    assert candidate(3) == 'fizz'
    assert candidate(5) == 'buzz'
    assert candidate(15) == 'fizzbuzz'
    assert candidate(7) == '7'
""",
    },
    {
        "task_id": "he_count_words",
        "category": "strings",
        "difficulty": "easy",
        "prompt": "Write `count_words(text)` returning the number of whitespace-separated words.",
        "entry_point": "count_words",
        "test": """
def check(candidate):
    assert candidate('') == 0
    assert candidate('hello world') == 2
    assert candidate('  a   b  ') == 2
""",
    },
    {
        "task_id": "he_unique_sorted",
        "category": "collections",
        "difficulty": "easy",
        "prompt": "Write `unique_sorted(items)` returning sorted unique elements preserving types.",
        "entry_point": "unique_sorted",
        "test": """
def check(candidate):
    assert candidate([3, 1, 2, 1, 3]) == [1, 2, 3]
    assert candidate([]) == []
""",
    },
    {
        "task_id": "he_two_sum_indices",
        "category": "algorithms",
        "difficulty": "medium",
        "prompt": (
            "Write `two_sum(nums, target)` returning indices i,j (i<j) where nums[i]+nums[j]==target, "
            "or None if impossible."
        ),
        "entry_point": "two_sum",
        "test": """
def check(candidate):
    assert candidate([2, 7, 11, 15], 9) == (0, 1)
    assert candidate([1, 2, 3], 10) is None
""",
    },
    {
        "task_id": "he_anagram",
        "category": "strings",
        "difficulty": "easy",
        "prompt": "Write `is_anagram(a, b)` ignoring case and whitespace.",
        "entry_point": "is_anagram",
        "test": """
def check(candidate):
    assert candidate('listen', 'silent') is True
    assert candidate('hello', 'bello') is False
""",
    },
    {
        "task_id": "he_flatten_once",
        "category": "collections",
        "difficulty": "medium",
        "prompt": "Write `flatten_once(nested)` flattening one level of nested lists.",
        "entry_point": "flatten_once",
        "test": """
def check(candidate):
    assert candidate([[1, 2], [3], []]) == [1, 2, 3]
    assert candidate([]) == []
""",
    },
    {
        "task_id": "he_run_length_encode",
        "category": "strings",
        "difficulty": "medium",
        "prompt": (
            "Write `run_length_encode(s)` returning a list of (char, count) tuples for consecutive runs."
        ),
        "entry_point": "run_length_encode",
        "test": """
def check(candidate):
    assert candidate('aaabbc') == [('a', 3), ('b', 2), ('c', 1)]
    assert candidate('') == []
""",
    },
    {
        "task_id": "he_binary_search",
        "category": "algorithms",
        "difficulty": "medium",
        "prompt": (
            "Write `binary_search(nums, target)` on sorted nums returning index or -1."
        ),
        "entry_point": "binary_search",
        "test": """
def check(candidate):
    assert candidate([1, 3, 5, 7], 5) == 2
    assert candidate([1, 3, 5, 7], 2) == -1
""",
    },
    {
        "task_id": "he_merge_dicts",
        "category": "collections",
        "difficulty": "easy",
        "prompt": (
            "Write `merge_dicts(a, b)` where later keys in b override a."
        ),
        "entry_point": "merge_dicts",
        "test": """
def check(candidate):
    assert candidate({'x': 1}, {'y': 2}) == {'x': 1, 'y': 2}
    assert candidate({'x': 1}, {'x': 3}) == {'x': 3}
""",
    },
    {
        "task_id": "he_clamp",
        "category": "math",
        "difficulty": "easy",
        "prompt": "Write `clamp(value, low, high)` returning value bounded to [low, high].",
        "entry_point": "clamp",
        "test": """
def check(candidate):
    assert candidate(5, 0, 10) == 5
    assert candidate(-1, 0, 10) == 0
    assert candidate(99, 0, 10) == 10
""",
    },
    {
        "task_id": "he_depth_sum",
        "category": "recursion",
        "difficulty": "hard",
        "prompt": (
            "Write `depth_sum(nested)` summing integers in a nested list structure, "
            "multiplying each integer by its nesting depth (top level depth=1)."
        ),
        "entry_point": "depth_sum",
        "test": """
def check(candidate):
    assert candidate([1, [2, [3]]]) == 14
    assert candidate([]) == 0
""",
    },
    {
        "task_id": "he_valid_brackets",
        "category": "stacks",
        "difficulty": "medium",
        "prompt": "Write `valid_brackets(s)` for brackets ()[]{} only.",
        "entry_point": "valid_brackets",
        "test": """
def check(candidate):
    assert candidate('()[]{}') is True
    assert candidate('(]') is False
    assert candidate('([{}])') is True
""",
    },
    {
        "task_id": "he_longest_unique_substr",
        "category": "algorithms",
        "difficulty": "hard",
        "prompt": (
            "Write `length_of_longest_substring(s)` returning length of longest substring "
            "without repeating characters."
        ),
        "entry_point": "length_of_longest_substring",
        "test": """
def check(candidate):
    assert candidate('abcabcbb') == 3
    assert candidate('bbbb') == 1
    assert candidate('') == 0
""",
    },
    {
        "task_id": "mbpp_list_sum",
        "category": "math",
        "difficulty": "easy",
        "prompt": "Write `list_sum(nums)` returning sum of a list of numbers.",
        "entry_point": "list_sum",
        "test": """
def check(candidate):
    assert candidate([1, 2, 3]) == 6
    assert candidate([]) == 0
""",
    },
    {
        "task_id": "mbpp_reverse_words",
        "category": "strings",
        "difficulty": "easy",
        "prompt": "Write `reverse_words(s)` reversing word order in a sentence.",
        "entry_point": "reverse_words",
        "test": """
def check(candidate):
    assert candidate('hello world') == 'world hello'
    assert candidate('a') == 'a'
""",
    },
    {
        "task_id": "mbpp_factorial",
        "category": "math",
        "difficulty": "easy",
        "prompt": "Write `factorial(n)` for n>=0.",
        "entry_point": "factorial",
        "test": """
def check(candidate):
    assert candidate(0) == 1
    assert candidate(5) == 120
""",
    },
    {
        "task_id": "mbpp_count_vowels",
        "category": "strings",
        "difficulty": "easy",
        "prompt": "Write `count_vowels(s)` counting aeiou regardless of case.",
        "entry_point": "count_vowels",
        "test": """
def check(candidate):
    assert candidate('Hello') == 2
    assert candidate('xyz') == 0
""",
    },
    {
        "task_id": "repair_off_by_one",
        "category": "debug",
        "difficulty": "medium",
        "prompt": (
            "Fix the buggy function `range_sum(n)` that should return sum(0..n) inclusive. "
            "Return only the corrected function."
        ),
        "entry_point": "range_sum",
        "starter_code": "def range_sum(n):\n    return sum(range(n))\n",
        "test": """
def check(candidate):
    assert candidate(3) == 6
    assert candidate(0) == 0
""",
    },
    {
        "task_id": "repair_wrong_compare",
        "category": "debug",
        "difficulty": "medium",
        "prompt": (
            "Fix `max_pair(a, b)` to return the larger of two numbers. Current code is wrong."
        ),
        "entry_point": "max_pair",
        "starter_code": "def max_pair(a, b):\n    return a if a < b else b\n",
        "test": """
def check(candidate):
    assert candidate(1, 2) == 2
    assert candidate(5, 3) == 5
""",
    },
]


def build_all_code_tasks() -> list[dict]:
    tasks: list[dict] = []
    for seed in _CODE_SEEDS:
        row = dict(seed)
        row.setdefault("language", "python")
        row.setdefault("requires_code_fence", True)
        row.setdefault("reward_mode", "binary")  # or "pass_rate" for partial credit
        tasks.append(row)
    return tasks


CODE_TASKS: list[dict] = build_all_code_tasks()
