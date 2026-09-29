from __future__ import annotations

import ast
import math
import re


def parse_num(value):
    value = str(value).replace(",", "").replace("_", "").strip()
    try:
        if "^" in value:
            base, exponent = value.split("^", 1)
            return float(base) ** float(exponent)
        if re.fullmatch(r"[+-]?[0-9]+[eE][+-]?[0-9]+", value):
            return float(value)
        return float(value)
    except (TypeError, ValueError):
        return None


def phrase_match_weight(text, weighted_terms):
    best = 0
    for term, weight in sorted(weighted_terms.items(), key=lambda item: (-len(str(item[0]).split()), -len(str(item[0])), str(item[0]).casefold())):
        pattern = r"(?<!\w)" + r"(?:\s+|[_-])+".join(re.escape(part) for part in str(term).split()) + r"(?!\w)"
        if re.search(pattern, text, flags=re.IGNORECASE):
            best = max(best, int(weight))
    return best


_NUMBER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])[+-]?[0-9][0-9,_]*(?:(?:\^|[eE])[+-]?[0-9]+)?(?![A-Za-z0-9_])")

def _constraint_values(text, patterns):
    matches = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            numbers = _NUMBER_TOKEN.findall(match.group(0))
            value = parse_num(numbers[-1]) if numbers else None
            matches.append(value)
    return matches


def _python_complexity(code, recursion_weight):
    tree = ast.parse(code)
    decision_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.BoolOp, ast.IfExp)
    f34 = 1 + sum(isinstance(node, decision_nodes) for node in ast.walk(tree))
    f34 += sum(isinstance(node, ast.comprehension) and bool(node.ifs) for node in ast.walk(tree))

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.loop_depth = 0
            self.nesting_sum = 0
            self.function_stack: list[str] = []
            self.recursions = 0

        def visit_FunctionDef(self, node):
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)

        def visit_For(self, node):
            self.loop_depth += 1
            self.nesting_sum += self.loop_depth
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_AsyncFor(self, node):
            self.visit_For(node)

        def visit_While(self, node):
            self.loop_depth += 1
            self.nesting_sum += self.loop_depth
            self.generic_visit(node)
            self.loop_depth -= 1

        def visit_Call(self, node):
            if self.function_stack and isinstance(node.func, ast.Name) and node.func.id == self.function_stack[-1]:
                self.recursions += 1
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return f34, visitor.nesting_sum + visitor.recursions * int(recursion_weight)


def _fallback_complexity(code, branch_keywords, loop_keywords, indent_width):
    f34 = 1
    for keyword in branch_keywords:
        f34 += len(re.findall(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", code, flags=re.IGNORECASE))
    brace_depth = 0
    nesting = 0
    uses_braces = "{" in code or "}" in code
    for line in code.splitlines():
        if uses_braces:
            for char in line:
                if char == "}":
                    brace_depth = max(0, brace_depth - 1)
                if char == "{":
                    brace_depth += 1
            depth = max(0, brace_depth - line.count("{"))
        else:
            expanded = line.expandtabs(indent_width)
            depth = len(expanded) - len(expanded.lstrip(" "))
            depth //= indent_width
        for keyword in loop_keywords:
            if re.search(r"\b" + re.escape(keyword) + r"\b", line, flags=re.IGNORECASE):
                nesting += max(0, depth)
    return f34, nesting


def extract(text, code_spans, alg_weights, constraint_patterns, recursion_weight=2, indent_width=4, branch_keywords=None, loop_keywords=None):
    # F30 is defined over nl_view + code_spans. The natural-language view
    # captures algorithm/DS names in prose while code_spans captures technique
    # names appearing inside code.
    algorithm_text = text
    if code_spans:
        algorithm_text += "\n" + "\n".join(code_spans)
    f30 = phrase_match_weight(algorithm_text, alg_weights)
    constraint_values = _constraint_values(text, constraint_patterns)
    numeric = [value for value in constraint_values if value is not None]
    f31 = math.log10(1 + max(numeric)) if numeric else math.nan
    f32 = len(constraint_values)
    if not code_spans:
        return {
            "F30": f30, "F31": f31, "F32": f32,
            "F33": math.nan, "F34": math.nan, "F35": math.nan,
            "code_parse_exact": 0,
        }
    code = "\n".join(code_spans)
    f33 = sum(1 for line in code.splitlines() if line.strip())
    try:
        f34, f35 = _python_complexity(code, recursion_weight)
        exact = 1
    except SyntaxError:
        f34, f35 = _fallback_complexity(
            code,
            branch_keywords or ["if", "else if", "elif", "for", "while", "case", "catch", "&&", "||", "?"],
            loop_keywords or ["for", "while", "foreach"],
            indent_width,
        )
        exact = 0
    return {"F30": f30, "F31": f31, "F32": f32, "F33": f33, "F34": f34, "F35": f35, "code_parse_exact": exact}
