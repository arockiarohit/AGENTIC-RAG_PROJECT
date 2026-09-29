import ast
import operator


ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _evaluate(node):
    """
    Safely evaluate an arithmetic AST.
    """

    # Numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value

        raise ValueError("Only numbers are allowed.")

    # Binary operations
    if isinstance(node, ast.BinOp):

        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator is not allowed.")

        left = _evaluate(node.left)
        right = _evaluate(node.right)

        return ALLOWED_OPERATORS[operator_type](left, right)

    # Unary operations: -10, +10
    if isinstance(node, ast.UnaryOp):

        operator_type = type(node.op)

        if operator_type not in ALLOWED_OPERATORS:
            raise ValueError("Operator is not allowed.")

        operand = _evaluate(node.operand)

        return ALLOWED_OPERATORS[operator_type](operand)

    raise ValueError("Invalid arithmetic expression.")


def safe_calculate(expression: str):
    """
    Safely calculate a mathematical expression.

    Example:
        125 * 48
        500 + 250
        1000 / 25
        2 ** 10
    """

    try:
        tree = ast.parse(expression, mode="eval")

        return _evaluate(tree.body)

    except Exception as e:
        raise ValueError(
            f"Unable to calculate expression: {expression}"
        ) from e