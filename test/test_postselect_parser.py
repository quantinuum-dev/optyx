
import random
import pytest
from optyx.utils.postselect_parser import compile_postselect

def run(expr, counts):
    return compile_postselect(expr)(counts)

@pytest.mark.parametrize("expr,counts,expected", [
    # Simple equality & inequality
    ("[0]==0", [0,1,2,0], [1]),
    ("[1]==1", [0,1,2,0], [1]),
    ("[2]==2", [0,1,2,0], [1]),
    ("[2]==1", [0,1,2,0], [0]),
    ("[0]>0",  [0,1,2,0], [0]),
    ("[1]>=1", [0,1,2,0], [1]),
    ("[2]<2",  [0,1,2,0], [0]),
    ("[2]<=2", [0,1,2,0], [1]),

    # Multi-mode sums
    ("[0,1]==1", [0,1,2,0], [1]),
    ("[0,1,2]==3", [0,1,2,0], [1]),
    ("[0,3]<1", [0,1,2,0], [1]),

    # AND
    ("[0,1]==1 & [2]==2", [0,1,2,0], [1]),
    ("[0,1]==1 & [2]==1", [0,1,2,0], [0]),

    # OR
    ("[0]==1 | [1]==1", [0,1,2,0], [1]),
    ("[0]==1 | [2]==1", [0,1,2,0], [0]),

    # XOR
    ("[0]==0 ^ [1]==1", [0,1,2,0], [0]),   # both true -> false
    ("[0]==1 ^ [1]==1", [0,1,2,0], [1]),   # false ^ true

    # NOT variants
    ("!([0]==1)", [0,1,2,0], [1]),
    ("not ([1]==1)", [0,1,2,0], [0]),
    ("NOT ([2]==1)", [0,1,2,0], [1]),

    # Parentheses
    ("([0]==0 & [1]==1) | [2]==0", [0,1,2,0], [1]),
    ("[0]==0 & ([1]==0 | [2]==0)", [0,1,2,0], [0]),

    # Textual logical operators
    ("[0,1]==1 and [2]==2", [0,1,2,0], [1]),
    ("[0,1]==2 or [2]==2", [0,1,2,0], [1]),
    ("[1]==1 xor [2]==2", [0,1,2,0], [0]),
    ("not [1]==1", [0,1,2,0], [0]),

    # Out-of-range modes treated as 0
    ("[10]==0", [0,1,2,0], [1]),
    ("[10,0]==0", [0,1,2,0], [1]),
    ("[10,2]>=2", [0,1,2,0], [1]),

    # More complex datasets
    ("[0,1,2]==2 & [3,4]>=4", [1,0,1,2,3], [1]),
    ("([0,1]==1) ^ ([2,3]==3)", [1,0,1,2,3], [0]),
    ("!([0]==1 ^ [1]==0) | [4]<3", [1,0,1,2,3], [1]),

    # All zeros
    ("[0,1,2]==0", [0,0,0], [1]),
    ("[0]>0 | [1]>0 | [2]>0", [0,0,0], [0]),
    ("!([0]==0 & [1]==0 & [2]==0)", [0,0,0], [0]),

    # All twos
    ("[0,1]>=4 & [2,3]>=4", [2,2,2,2], [1]),
    ("[0,1,2,3]==8", [2,2,2,2], [1]),
    ("[0,1]<4 ^ [2,3]<4", [2,2,2,2], [0]),

    # Mixed parens & ops
    ("( [0,1]==2 & [2,3]==0 ) | ( [0]==1 ^ [1]==1 )", [1,1,0,0], [1]),
    ("([0]==1 & [1]==1) ^ ([2]==0 & [3]==0)", [1,1,0,0], [0]),
    ("([0]==2 | [1]==2) & !([2]==0 & [3]==0)", [1,1,0,0], [0]),

    # <=, >= and more nesting
    ("[0,1]<=2 & [2]>=1", [0,2,1], [1]),
    ("[0,2]<1 | [1]>2", [0,2,1], [0]),
    ("([0,1]==2) ^ !([2]==0)", [0,2,1], [0]),

    # Out-of-range again
    ("[0,10]==3 & [3]==1", [3,0,0,1], [1]),
    ("[1,2,5]==0", [3,0,0,1], [1]),
    ("!([0]==3) | [2]==1", [3,0,0,1], [0]),

    # textual mix
    ("[0]==1 and ([1]==0 or [2]==0)", [1,0,0], [1]),
    ("not ([0]==1) or [1]==0", [1,0,0], [1]),
    ("[0]==0 xor [1]==0", [1,0,0], [1]),

    # Larger list combos
    ("[0,2,4]==6 & ([1,3]==0 | [0,1]<3)", [2,0,2,0,2], [1]),
    ("([0,1]==2) ^ ([2,3]==2) ^ ([4]==2)", [2,0,2,0,2], [1]),
    ("!([0]==2 & [2]==2) | [1,3]>=1", [2,0,2,0,2], [0]),
])
def test_explicit(expr, counts, expected):
    assert run(expr, counts) == expected

def random_condition(max_mode: int) -> str:
    k = random.randint(1, min(3, max_mode+1))
    modes = sorted(random.sample(range(max_mode+1), k))
    op = random.choice(["==", ">=", "<=", ">", "<"])
    rhs = random.randint(0, 3)
    return f"[{','.join(map(str,modes))}] {op} {rhs}"

def random_group(max_mode: int) -> str:
    # group uses a single operator among conditions, as per spec
    n_conds = random.randint(1, 3)
    conds = [random_condition(max_mode) for _ in range(n_conds)]
    op = random.choice(["&", "|", "^"])
    s = f" {op} ".join(conds)
    return f"({s})" if random.random() < 0.9 else s

def random_expression(max_mode: int) -> str:
    # combine groups with explicit parentheses to avoid ambiguity
    n_groups = random.randint(1, 3)
    groups = []
    for _ in range(n_groups):
        g = random_group(max_mode)
        if random.random() < 0.4:
            g = "!" + f"({g})"
        groups.append(g)
    expr = groups[0]
    for g in groups[1:]:
        op = random.choice(["&", "|", "^"])
        expr = f"({expr}) {op} ({g})"
    return expr

def eval_reference(expr: str, counts):
    # a simple reference evaluator by evaluating each condition and using Python's boolean ops
    import re
    e = expr
    e = re.sub(r"\bAND\b|\band\b", "&", e)
    e = re.sub(r"\bOR\b|\bor\b", "|", e)
    e = re.sub(r"\bXOR\b|\bxor\b", "^", e)
    e = re.sub(r"\bNOT\b|\bnot\b", "!", e)

    cond_pat = re.compile(r"\[(?P<modes>\s*\d+(?:\s*,\s*\d+)*\s*)\]\s*(?P<op>==|>=|<=|>|<)\s*(?P<rhs>\d+)")
    def cond_repl(m):
        modes = [int(x.strip()) for x in m.group("modes").split(",")]
        rhs = int(m.group("rhs"))
        op = m.group("op")
        s = sum(counts[i] if i < len(counts) else 0 for i in modes)
        val = {
            "==": s == rhs,
            ">=": s >= rhs,
            "<=": s <= rhs,
            ">":  s >  rhs,
            "<":  s <  rhs,
        }[op]
        return f"({str(val)})"
    prev = None
    while prev != e:
        prev = e
        e = cond_pat.sub(cond_repl, e)
    e = re.sub(r"!", " not ", e)
    return [1] if bool(eval(e, {"__builtins__": {}}, {})) else [0]

@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_randomized(seed):
    random.seed(seed)
    for _ in range(25): 
        max_mode = random.randint(2, 6)
        counts = [random.randint(0, 3) for _ in range(max_mode+1)]
        expr = random_expression(max_mode)
        assert run(expr, counts) == eval_reference(expr, counts)
