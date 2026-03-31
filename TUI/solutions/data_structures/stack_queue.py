"""
NAME: Stack and Queue Patterns
TAGS: stack, queue, design, string
DESCRIPTION: Classic stack and queue design problems: balanced parentheses, min-stack,
             queue using stacks, and stack with O(1) minimum.
             These patterns appear frequently in system design and algorithm rounds.
COMPLEXITY: Time O(1) amortized per operation, Space O(n)
"""

from collections import deque
from typing import Optional, List


# ─── Valid Parentheses ────────────────────────────────────────────────────────
# Check if brackets are balanced and properly nested.

def is_valid_parentheses(s: str) -> bool:
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for ch in s:
        if ch in mapping:
            top = stack.pop() if stack else '#'
            if mapping[ch] != top:
                return False
        else:
            stack.append(ch)

    return not stack

# Example:
# "()[]{}"   -> True
# "([)]"     -> False
# "{[]}"     -> True


# ─── Minimum Remove to Make Valid Parentheses ─────────────────────────────────
# Remove minimum brackets to make the string valid.

def min_remove_to_make_valid(s: str) -> str:
    stack = []   # stores indices of unmatched '('
    to_remove = set()

    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()
            else:
                to_remove.add(i)

    to_remove |= set(stack)
    return ''.join(ch for i, ch in enumerate(s) if i not in to_remove)

# Example:
# "lee(t(c)o)de)" -> "lee(t(c)o)de"


# ─── Min Stack ────────────────────────────────────────────────────────────────
# Stack supporting push, pop, top, and getMin all in O(1).
# Strategy: store (value, current_min) pairs.

class MinStack:
    def __init__(self):
        self.stack = []  # stores (value, min_so_far)

    def push(self, val: int) -> None:
        current_min = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def get_min(self) -> int:
        return self.stack[-1][1]

# Example:
# ms = MinStack()
# ms.push(-2); ms.push(0); ms.push(-3)
# ms.get_min() -> -3
# ms.pop()
# ms.get_min() -> -2


# ─── Min Stack (space-optimized, auxiliary stack variant) ─────────────────────
# Only push to min_stack when a new minimum is encountered.

class MinStackAux:
    def __init__(self):
        self.stack = []
        self.min_stack = []  # only pushed when new min found

    def push(self, val: int) -> None:
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def get_min(self) -> int:
        return self.min_stack[-1]


# ─── Queue Using Two Stacks ───────────────────────────────────────────────────
# Amortized O(1) enqueue and dequeue.
# Push to in-stack; when dequeuing, if out-stack is empty, dump in-stack into it.

class QueueUsingStacks:
    def __init__(self):
        self.in_stack = []   # for enqueue
        self.out_stack = []  # for dequeue

    def enqueue(self, val: int) -> None:
        self.in_stack.append(val)

    def dequeue(self) -> int:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        if not self.out_stack:
            raise IndexError("dequeue from empty queue")
        return self.out_stack.pop()

    def peek(self) -> int:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1]

    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack

# Example:
# q = QueueUsingStacks()
# q.enqueue(1); q.enqueue(2)
# q.dequeue() -> 1


# ─── Stack Using Two Queues ───────────────────────────────────────────────────
# Push is O(n): rotate queue so new element is at front.

class StackUsingQueues:
    def __init__(self):
        self.q = deque()

    def push(self, val: int) -> None:
        self.q.append(val)
        # Rotate: move all previous elements behind the new one
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return not self.q


# ─── Evaluate Reverse Polish Notation (RPN) ───────────────────────────────────

def eval_rpn(tokens: List[str]) -> int:
    stack = []
    ops = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b,
        '/': lambda a, b: int(a / b),  # truncate toward zero
    }
    for token in tokens:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))
    return stack[0]

# Example:
# ["2","1","+","3","*"] -> 9   ((2+1)*3)
# ["4","13","5","/","+"] -> 6  (4 + 13/5)


# ─── Basic Calculator II (with +, -, *, /) ────────────────────────────────────
# Parse and evaluate expression string without parentheses.

def calculate_ii(s: str) -> int:
    stack = []
    num = 0
    sign = '+'

    for i, ch in enumerate(s):
        if ch.isdigit():
            num = num * 10 + int(ch)

        if (ch in '+-*/' or i == len(s) - 1) and ch != ' ':
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                stack.append(int(stack.pop() / num))  # truncate toward zero
            sign = ch
            num = 0

    return sum(stack)

# Example:
# "3+2*2"    -> 7
# " 3/2 "    -> 1
# " 3+5 / 2" -> 5


# ─── Basic Calculator I (with +, -, parentheses) ─────────────────────────────

def calculate_i(s: str) -> int:
    stack = []
    num = 0
    result = 0
    sign = 1  # +1 or -1

    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        elif ch == '+':
            result += sign * num
            num = 0
            sign = 1
        elif ch == '-':
            result += sign * num
            num = 0
            sign = -1
        elif ch == '(':
            # Save current result and sign on stack
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif ch == ')':
            result += sign * num
            num = 0
            result *= stack.pop()   # restore sign before '('
            result += stack.pop()   # restore result before '('

    return result + sign * num

# Example:
# "1 + 1"         -> 2
# " 2-1 + 2 "     -> 3
# "(1+(4+5+2)-3)+(6+8)" -> 23


# ─── Decode String (k[encoded_string]) ───────────────────────────────────────

def decode_string(s: str) -> str:
    stack = []
    curr_str = ""
    curr_num = 0

    for ch in s:
        if ch.isdigit():
            curr_num = curr_num * 10 + int(ch)
        elif ch == '[':
            stack.append((curr_str, curr_num))
            curr_str, curr_num = "", 0
        elif ch == ']':
            prev_str, num = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += ch

    return curr_str

# Example:
# "3[a]2[bc]"    -> "aaabcbc"
# "3[a2[c]]"     -> "accaccacc"
# "2[abc]3[cd]ef" -> "abcabccdcdcdef"


# ─── Largest Rectangle using Stack (see monotonic_stack.py) ──────────────────
# (Refer to monotonic_stack.py for the full histogram implementation.)


# ─── Circular Queue (Ring Buffer) ─────────────────────────────────────────────
# Fixed-size queue with O(1) enqueue and dequeue. Useful for BFS.

class CircularQueue:
    def __init__(self, capacity: int):
        self.buf = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0
        self.capacity = capacity

    def enqueue(self, val) -> bool:
        if self.size == self.capacity:
            return False
        self.buf[self.tail] = val
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1
        return True

    def dequeue(self):
        if self.size == 0:
            raise IndexError("dequeue from empty queue")
        val = self.buf[self.head]
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return val

    def front(self):
        if self.size == 0:
            return -1
        return self.buf[self.head]

    def rear(self):
        if self.size == 0:
            return -1
        return self.buf[(self.tail - 1) % self.capacity]

    def is_empty(self) -> bool:
        return self.size == 0

    def is_full(self) -> bool:
        return self.size == self.capacity


if __name__ == "__main__":
    assert is_valid_parentheses("()[]{}")
    assert not is_valid_parentheses("([)]")
    assert is_valid_parentheses("{[]}")

    ms = MinStack()
    ms.push(-2); ms.push(0); ms.push(-3)
    assert ms.get_min() == -3
    ms.pop()
    assert ms.get_min() == -2

    q = QueueUsingStacks()
    q.enqueue(1); q.enqueue(2)
    assert q.dequeue() == 1

    assert eval_rpn(["2","1","+","3","*"]) == 9
    assert calculate_ii("3+2*2") == 7
    assert calculate_i("(1+(4+5+2)-3)+(6+8)") == 23
    assert decode_string("3[a2[c]]") == "accaccacc"
    print("All tests passed.")
