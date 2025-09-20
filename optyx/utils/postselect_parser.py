
import re
from typing import List


class _Tok:
    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value


def _tokenize(s: str):
    s = s.strip()
    # textual logical operators to symbols
    s = re.sub(r"\bAND\b|\band\b", "&", s)
    s = re.sub(r"\bOR\b|\bor\b", "|", s)
    s = re.sub(r"\bXOR\b|\bxor\b", "^", s)
    s = re.sub(r"\bNOT\b|\bnot\b", "!", s)

    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "[](),&|^!":
            tokens.append(_Tok(c, c))
            i += 1
            continue
        if c in "<>=":
            if s.startswith("==", i):
                tokens.append(_Tok("OP", "=="))
                i += 2
                continue
            if s.startswith(">=", i):
                tokens.append(_Tok("OP", ">="))
                i += 2
                continue
            if s.startswith("<=", i):
                tokens.append(_Tok("OP", "<="))
                i += 2
                continue
            if c == ">":
                tokens.append(_Tok("OP", ">"))
                i += 1
                continue
            if c == "<":
                tokens.append(_Tok("OP", "<"))
                i += 1
                continue
            raise ValueError(f"Unexpected operator at position {i}")
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(_Tok("INT", s[i:j]))
            i = j
            continue
        if c == ",":
            tokens.append(_Tok(",", ","))
            i += 1
            continue
        raise ValueError(f"Unexpected character {c!r} at position {i}")
    tokens.append(_Tok("EOF", ""))
    return tokens


class _Cond:
    def __init__(self, modes: List[int], op: str, rhs: int):
        self.modes = modes
        self.op = op
        self.rhs = rhs

    def eval(self, counts: List[int]) -> bool:
        s = 0
        for m in self.modes:
            if m < 0:
                raise ValueError("Mode indices must be non-negative")
            s += counts[m] if m < len(counts) else 0
        if self.op == "==":
            return s == self.rhs
        if self.op == ">=":
            return s >= self.rhs
        if self.op == "<=":
            return s <= self.rhs
        if self.op == ">":
            return s > self.rhs
        if self.op == "<":
            return s < self.rhs
        raise ValueError(f"Unknown operator {self.op!r}")


class _Not:
    def __init__(self, x): self.x = x
    def eval(self, counts: List[int]) -> bool: return not self.x.eval(counts)


class _Bin:
    def __init__(self, op: str, xs):
        self.op = op
        self.xs = xs

    def eval(self, counts: List[int]) -> bool:
        if self.op == "&":
            out = True
            for x in self.xs:
                out = out and x.eval(counts)
            return out
        if self.op == "|":
            out = False
            for x in self.xs:
                out = out or x.eval(counts)
            return out
        if self.op == "^":
            acc = False
            for x in self.xs:
                acc ^= x.eval(counts)
            return acc
        raise ValueError(f"Unknown logical op {self.op!r}")


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def _peek(self):
        return self.toks[self.i]

    def _eat(self, kind):
        t = self._peek()
        if (
            (isinstance(kind, tuple) and
                t.kind in kind) or
                t.kind == kind or t.value == kind
        ):
            self.i += 1
            return t
        raise ValueError(f"Expected {kind}, got {t.kind}:{t.value}")

    def parse(self):
        node = self._parse_or()
        if self._peek().kind != "EOF":
            raise ValueError("Unexpected trailing input")
        return node

    # Precedence: ! > & > ^ > |
    def _parse_or(self):
        left = self._parse_xor()
        xs = [left]
        while self._peek().value == "|":
            self._eat("|")
            xs.append(self._parse_xor())
        return xs[0] if len(xs) == 1 else _Bin("|", xs)

    def _parse_xor(self):
        left = self._parse_and()
        xs = [left]
        while self._peek().value == "^":
            self._eat("^")
            xs.append(self._parse_and())
        return xs[0] if len(xs) == 1 else _Bin("^", xs)

    def _parse_and(self):
        left = self._parse_not()
        xs = [left]
        while self._peek().value == "&":
            self._eat("&")
            xs.append(self._parse_not())
        return xs[0] if len(xs) == 1 else _Bin("&", xs)

    def _parse_not(self):
        if self._peek().value == "!":
            self._eat("!")
            return _Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self):
        t = self._peek()
        if t.value == "(":
            self._eat("(")
            e = self._parse_or()
            self._eat(")")
            return e
        if t.value == "[":
            self._eat("[")
            modes = [int(self._eat("INT").value)]
            while self._peek().value == ",":
                self._eat(",")
                modes.append(int(self._eat("INT").value))
            self._eat("]")
            op = self._eat("OP").value
            rhs = int(self._eat("INT").value)
            if rhs < 0:
                raise ValueError("Photon count must be a non-negative integer")
            return _Cond(modes, op, rhs)
        raise ValueError(f"Expected condition or '(', got {t.kind}:{t.value}")


def compile_postselect(expression: str):
    """
    Compile a PostSelect expression into a callable:
        f(counts) -> [1] if expression is satisfied on 'counts', otherwise [0].

    - counts: list of non-negative integers (photons per mode; index = mode id)
    - Missing modes referenced by the expression are treated as 0.
    """
    ast = _Parser(_tokenize(expression)).parse()

    def predicate(counts):
        ok = ast.eval(counts)
        return [1] if ok else [0]
    return predicate
