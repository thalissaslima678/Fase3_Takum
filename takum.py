import math
from fractions import Fraction
from dataclasses import dataclass

@dataclass
class Fields:
    n: int
    bits: int
    kind: str
    S: int = 0
    D: int = 0
    R: int = 0
    r: int = 0
    C: int = 0
    F: int = 0
    p: int = 0
    c: int = 0
    F_ext: int = 0
    p_ext: int = 0

def decode_fields(bits: int, n: int) -> Fields:
    bits &= (1 << n) - 1
    if bits == 0:
        return Fields(n, bits, "zero")
    if bits == 1 << (n - 1):
        return Fields(n, bits, "nar", S=1)
    w = max(n, 12)
    pad = bits << (w - n)         
    S = (pad >> (w - 1)) & 1
    D = (pad >> (w - 2)) & 1
    R = (pad >> (w - 5)) & 0b111
    r = R if D else 7 - R
    C = (pad >> (w - 5 - r)) & ((1 << r) - 1)
    p_ext = w - 5 - r
    F_ext = pad & ((1 << p_ext) - 1)
    p = max(0, n - 5 - r)
    F = F_ext >> (p_ext - p)
    c = (C + (1 << r) - 1) if D else (C - (1 << (r + 1)) + 1)
    return Fields(n, bits, "normal", S, D, R, r, C, F, p, c, F_ext, p_ext)

def unpack(bits: int, n: int):
    f = decode_fields(bits, n)
    if f.kind == "zero":
        return (0, 0, 0)
    if f.kind == "nar":
        return None
    if f.S == 0:
        return (1, (1 << f.p_ext) | f.F_ext, f.c - f.p_ext)
    return (-1, (2 << f.p_ext) - f.F_ext, -(f.c + 1) - f.p_ext)

C_MAX = 254
C_MIN = -255

def encode_magnitude(M: int, e: int, n: int) -> int:
    maxpos = (1 << (n - 1)) - 1
    if M == 0: return 0
    pf = M.bit_length() - 1
    c = e + pf
    if c > C_MAX:
        return maxpos
    if c < C_MIN:
        return 1
    F = M - (1 << pf)
    if c >= 0:
        D = 1
        r = (c + 1).bit_length() - 1
        C = c - (1 << r) + 1
    else:
        D = 0
        r = (-c).bit_length() - 1
        C = c + (1 << (r + 1)) - 1
    R = r if D else 7 - r
    v = (((((D << 3) | R) << r) | C) << pf) | F
    length = 4 + r + pf
    budget = n - 1
    if length <= budget:
        keep = v << (budget - length)
    else:
        shift = length - budget
        keep = v >> shift
        rem = v & ((1 << shift) - 1)
        half = 1 << (shift - 1)
        if rem > half or (rem == half and (keep & 1)):
            keep += 1
    return max(1, min(keep, maxpos))

def encode(sign: int, M: int, e: int, n: int) -> int:
    if M == 0:
        return 0
    pos = encode_magnitude(M, e, n)
    return pos if sign > 0 else (-pos) & ((1 << n) - 1)

class TakumFormat:
    def __init__(self, n):
        self.n = n
        self.mask = (1 << n) - 1
        self.NAR = 1 << (n - 1)

    def decode(self, bits):
        u = unpack(bits, self.n)
        if u is None: return None 
        s, M, e = u
        return s * M * (2.0 ** e)

    def add(self, a, b):
        ua, ub = unpack(a, self.n), unpack(b, self.n)
        if ua is None or ub is None: return self.NAR
        (sa, Ma, ea), (sb, Mb, eb) = ua, ub
        if Ma == 0: return b & self.mask
        if Mb == 0: return a & self.mask
        e = min(ea, eb)
        v = sa * (Ma << (ea - e)) + sb * (Mb << (eb - e))
        if v == 0: return 0
        return encode(1 if v > 0 else -1, abs(v), e, self.n)

    def neg(self, a):
        return (-a) & self.mask

    def sub(self, a, b):
        return self.add(a, self.neg(b))

    def mul(self, a, b):
        ua, ub = unpack(a, self.n), unpack(b, self.n)
        if ua is None or ub is None: return self.NAR
        (sa, Ma, ea), (sb, Mb, eb) = ua, ub
        if Ma == 0 or Mb == 0: return 0
        return encode(sa * sb, Ma * Mb, ea + eb, self.n)

    def from_fraction(self, x: Fraction) -> int:
        x = Fraction(x)
        if x == 0: return 0
        num, den = abs(x.numerator), x.denominator
        k = max(0, den.bit_length() + self.n + 600)
        q, rem = divmod(num << k, den)
        if rem:
            q = (q << 1) | 1
            k += 1
        return encode(1 if x > 0 else -1, q, -k, self.n)

    def from_float(self, x: float) -> int:
        if math.isnan(x) or math.isinf(x): return self.NAR
        return self.from_fraction(Fraction(x))

    def to_fraction(self, bits: int) -> Fraction:
        u = unpack(bits, self.n)
        if u is None: return None
        s, M, e = u
        return Fraction(s * M * (2**e) if e >= 0 else s * M, 2**(-e) if e < 0 else 1)

    def to_float(self, bits: int) -> float:
        val = self.to_fraction(bits)
        if val is None: return float('nan')
        return float(val)

    def to_signed(self, bits: int) -> int:
        return bits if bits < (1 << (self.n - 1)) else bits - (1 << self.n)

FORMATS = {
    8: TakumFormat(8),
    16: TakumFormat(16),
    32: TakumFormat(32),
    64: TakumFormat(64)
}