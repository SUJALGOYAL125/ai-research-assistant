# calculator_tool.py

import ast                                # lets us safely read a math expression, piece by piece
import operator                           # gives us real functions for +, -, *, /, **
from langchain_core.tools import tool     # turns our function into an agent tool

# Step 1: map each math symbol to a real function
# For example, "+" maps to operator.add, which does a + b
OPS = {
    ast.Add: operator.add,     # +
    ast.Sub: operator.sub,     # -
    ast.Mult: operator.mul,    # *
    ast.Div: operator.truediv, # /
    ast.Pow: operator.pow,     # **
}


# Step 2: this function solves the expression one small piece at a time
def evaluate(node):
    # If it's just a plain number, like 5, return it as-is
    if isinstance(node, ast.Constant):
        return node.value

    # If it's an operation, like "3 + 4"
    if isinstance(node, ast.BinOp):
        op_func = OPS[type(node.op)]              # find the right function (+, -, *, /, **)
        return op_func(evaluate(node.left), evaluate(node.right))  # solve both sides, then combine

    # Anything else (not a number or basic math) gets refused
    raise ValueError("Only numbers and + - * / ** are allowed")


# Step 3: this is the actual tool our AI agent will call
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression like '2 + 2', '(3 + 4) * 2', or '6 ** 2' for exponents.
    Use ** for powers/exponents, not ^."""
    try:
        tree = ast.parse(expression, mode="eval")  # read the text as math, not as runnable code
        result = evaluate(tree.body)                # solve it using our safe function
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:
        return "Error: invalid expression"


# Step 4: quick test to make sure it works
if __name__ == "__main__":
    print(calculator.invoke({"expression": "2 + 2"}))        # should print 4
    print(calculator.invoke({"expression": "(3 + 4) * 2"}))  # should print 14
    print(calculator.invoke({"expression": "10 / 0"}))       # should print an error