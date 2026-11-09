"""
rvv_takum.py - Fase 2: Extensao Zvtakum sobre um modelo simplificado da RVV.

Modelo de maquina (subconjunto da RISC-V Vector Extension):
  * 32 registradores vetoriais de VLEN bits (um unico bytearray contiguo,
    elementos little-endian; grupos de registradores LMUL = 1, 2, 4, 8).
  * CSRs vetoriais: vl, vtype (SEW, LMUL, vta, vma), vlenb.
    Zvtakum NAO adiciona CSRs e dispensa frm/fflags (um unico modo de
    arredondamento, sem flags de excecao: ver takum.py).
  * Mascaramento por v0 (bit i de v0 habilita o elemento i).
  * Politica tail/mask "undisturbed" (permitida pela politica "agnostic").

Instrucoes Zvtakum (SEW seleciona o formato: e8->T8, e16->T16, ...):
  vtadd.vv  vd, vs2, vs1[, v0.t]   vd[i] = vs2[i] + vs1[i]
  vtsub.vv  vd, vs2, vs1[, v0.t]   vd[i] = vs2[i] - vs1[i]
  vtmul.vv  vd, vs2, vs1[, v0.t]   vd[i] = vs2[i] * vs1[i]
  vtdot.vv  vd, vs2, vs1[, v0.t]   vd[0] (EEW=2*SEW) += sum_i vs2[i]*vs1[i]
                                    soma EXATA, um unico arredondamento
  vncvt.t.t vd, vs2[, v0.t]        vd[i] (SEW) = round(vs2[i] (2*SEW))

Codificacao proposta: opcode custom-2 (0x5B, reservado pela especificacao
RISC-V para extensoes de fabricante), mesmo layout de campos do OP-V:
  | funct6 | vm | vs2 | vs1 | funct3=000 | vd | 1011011 |
Os valores de funct6 sao arbitrarios dentro do espaco custom.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from takum import TakumFormat, FORMATS, unpack, encode

FMT = dict(FORMATS)
FMT[64] = TakumFormat(64)            # acumulador do vtdot.vv em SEW = 32

OPC_V = 0b1010111
OPC_CUSTOM2 = 0b1011011
F3_TAKUM = 0b000
F3_VSETVLI = 0b111

# mnemonico -> (funct6, formato)   'vv': vd, vs2, vs1   'v': vd, vs2 (vs1 = 0)
INSTR = {
    "vtadd.vv": (0b000000, "vv"),
    "vtsub.vv": (0b000010, "vv"),
    "vtmul.vv": (0b100100, "vv"),
    "vtdot.vv": (0b111001, "vv"),
    "vncvt.t.t": (0b010010, "v"),
}
BY_FUNCT6 = {f6: (mn, kind) for mn, (f6, kind) in INSTR.items()}
SEW_ENC = {8: 0, 16: 1, 32: 2, 64: 3}
LMUL_ENC = {1: 0, 2: 1, 4: 2, 8: 3}
CSRS_ADDED_BY_ZVTAKUM = ()          # nenhum
FP_CSRS_REQUIRED = ()               # sem frm / fflags


class IllegalInstruction(Exception):
    pass


# ----------------------------------------------------------------------------
# Assembler / disassembler
# ----------------------------------------------------------------------------
_REG = re.compile(r"^([vx])(\d+)$")


def _reg(tok: str, kind: str) -> int:
    m = _REG.match(tok.strip())
    if not m or m.group(1) != kind or int(m.group(2)) > 31:
        raise ValueError(f"registrador invalido: {tok!r}")
    return int(m.group(2))


def _word(funct6, vm, vs2, vs1, funct3, vd, opcode):
    return (funct6 << 26) | (vm << 25) | (vs2 << 20) | (vs1 << 15) | (funct3 << 12) | (vd << 7) | opcode


def assemble(line: str):
    """Monta uma linha de assembly em uma palavra de 32 bits (None se vazia)."""
    line = line.split("#")[0].strip()
    if not line:
        return None
    parts = line.split(None, 1)
    mn = parts[0]
    ops = [o.strip() for o in parts[1].split(",")] if len(parts) > 1 else []
    if mn == "vsetvli":
        rd, rs1 = _reg(ops[0], "x"), _reg(ops[1], "x")
        sew = int(ops[2][1:]) if ops[2].startswith("e") else None
        lmul = int(ops[3][1:]) if ops[3].startswith("m") else None
        if sew not in SEW_ENC or lmul not in LMUL_ENC:
            raise ValueError("vsetvli: SEW/LMUL nao suportado")
        vta = 1 if "ta" in ops[4:] else 0
        vma = 1 if "ma" in ops[4:] else 0
        zimm = (vma << 7) | (vta << 6) | (SEW_ENC[sew] << 3) | LMUL_ENC[lmul]
        return (zimm << 20) | (rs1 << 15) | (F3_VSETVLI << 12) | (rd << 7) | OPC_V
    if mn not in INSTR:
        raise ValueError(f"instrucao desconhecida: {mn}")
    f6, kind = INSTR[mn]
    vm = 1
    if ops and ops[-1] == "v0.t":
        vm, ops = 0, ops[:-1]
    if kind == "vv":
        if len(ops) != 3:
            raise ValueError(f"{mn} espera vd, vs2, vs1")
        vd, vs2, vs1 = (_reg(o, "v") for o in ops)
    else:
        if len(ops) != 2:
            raise ValueError(f"{mn} espera vd, vs2")
        vd, vs2, vs1 = _reg(ops[0], "v"), _reg(ops[1], "v"), 0
    return _word(f6, vm, vs2, vs1, F3_TAKUM, vd, OPC_CUSTOM2)


def disassemble(word: int) -> str:
    opcode, funct3 = word & 0x7F, (word >> 12) & 7
    rd, rs1 = (word >> 7) & 31, (word >> 15) & 31
    if opcode == OPC_V and funct3 == F3_VSETVLI and not (word >> 31):
        zimm = (word >> 20) & 0x7FF
        sew = {v: k for k, v in SEW_ENC.items()}.get((zimm >> 3) & 7)
        lmul = {v: k for k, v in LMUL_ENC.items()}.get(zimm & 7)
        ta = "ta" if (zimm >> 6) & 1 else "tu"
        ma = "ma" if (zimm >> 7) & 1 else "mu"
        return f"vsetvli x{rd}, x{rs1}, e{sew}, m{lmul}, {ta}, {ma}"
    if opcode == OPC_CUSTOM2 and funct3 == F3_TAKUM and (word >> 26) in BY_FUNCT6:
        mn, kind = BY_FUNCT6[word >> 26]
        vm, vs2 = (word >> 25) & 1, (word >> 20) & 31
        text = f"{mn} v{rd}, v{vs2}" + (f", v{rs1}" if kind == "vv" else "")
        return text + ("" if vm else ", v0.t")
    raise IllegalInstruction(f"palavra desconhecida: 0x{word:08X}")


# ----------------------------------------------------------------------------
# Nucleo aritmetico do vtdot (soma exata de produtos, um unico arredondamento)
# ----------------------------------------------------------------------------
def dot_exact(xs, ys, acc_bits: int, n_in: int, n_out: int) -> int:
    """acc + sum(x_i * y_i), exato, arredondado uma vez para T(n_out)."""
    nar = 1 << (n_out - 1)
    terms = []
    ua = unpack(acc_bits, n_out)
    if ua is None:
        return nar
    if ua[1]:
        terms.append(ua)
    for x, y in zip(xs, ys):
        ux, uy = unpack(x, n_in), unpack(y, n_in)
        if ux is None or uy is None:
            return nar
        (sx, Mx, ex), (sy, My, ey) = ux, uy
        if Mx and My:
            terms.append((sx * sy, Mx * My, ex + ey))
    if not terms:
        return 0
    e = min(t[2] for t in terms)
    v = sum(s * (M << (ee - e)) for s, M, ee in terms)
    return 0 if v == 0 else encode(1 if v > 0 else -1, abs(v), e, n_out)


@dataclass
class Stats:
    instructions: int = 0
    element_ops: int = 0
    by_mnemonic: dict = field(default_factory=dict)

    def record(self, mn: str, elems: int = 0):
        self.instructions += 1
        self.element_ops += elems
        self.by_mnemonic[mn] = self.by_mnemonic.get(mn, 0) + 1


# ----------------------------------------------------------------------------
# Maquina vetorial
# ----------------------------------------------------------------------------
class VectorMachine:
    def __init__(self, vlen: int = 128):
        if vlen < 64 or vlen & (vlen - 1):
            raise ValueError("VLEN deve ser potencia de 2 e >= 64")
        self.vlen, self.vlenb = vlen, vlen // 8
        self.mem = bytearray(32 * self.vlenb)
        self.x = [0] * 32                    # registradores escalares
        self.vl = 0
        self.sew, self.lmul, self.vta, self.vma = 8, 1, 0, 0
        self.vill = True
        self.stats = Stats()

    # --- acesso ao banco de registradores ---------------------------------
    def _addr(self, vreg, sew, i):
        return vreg * self.vlenb + i * (sew // 8)

    def read_elem(self, vreg, sew, i):
        a, nb = self._addr(vreg, sew, i), sew // 8
        if a + nb > len(self.mem):
            raise IllegalInstruction("acesso fora do banco de registradores")
        return int.from_bytes(self.mem[a:a + nb], "little")

    def write_elem(self, vreg, sew, i, value):
        a, nb = self._addr(vreg, sew, i), sew // 8
        if a + nb > len(self.mem):
            raise IllegalInstruction("acesso fora do banco de registradores")
        self.mem[a:a + nb] = (value & ((1 << sew) - 1)).to_bytes(nb, "little")

    def read_vec(self, vreg, sew, n=None):
        n = self.vl if n is None else n
        return [self.read_elem(vreg, sew, i) for i in range(n)]

    def write_vec(self, vreg, sew, values):
        """Equivale a vle<sew>.v (cargas/stores sao independentes do formato)."""
        for i, v in enumerate(values):
            self.write_elem(vreg, sew, i, v)

    def write_mask(self, bits):
        """Escreve a mascara em v0 (bits[i] -> elemento i)."""
        for i in range(self.vlen):
            self.mem[i // 8] &= ~(1 << (i % 8)) & 0xFF
        for i, b in enumerate(bits):
            if b:
                self.mem[i // 8] |= 1 << (i % 8)

    def _active(self, i, vm):
        return True if vm else bool((self.mem[i // 8] >> (i % 8)) & 1)

    # --- configuracao ---------------------------------------------------------
    @property
    def vlmax(self):
        return self.lmul * self.vlen // self.sew

    @property
    def vtype(self):
        if self.vill:
            return 1 << 63
        return (self.vma << 7) | (self.vta << 6) | (SEW_ENC[self.sew] << 3) | LMUL_ENC[self.lmul]

    def vsetvli(self, avl, sew, lmul, vta=0, vma=0):
        if sew not in SEW_ENC or lmul not in LMUL_ENC:
            self.vill, self.vl = True, 0
        else:
            self.vill = False
            self.sew, self.lmul, self.vta, self.vma = sew, lmul, vta, vma
            self.vl = min(avl, self.vlmax)
        self.stats.record("vsetvli")
        return self.vl

    # --- verificacoes --------------------------------------------------------
    def _require(self):
        if self.vill:
            raise IllegalInstruction("vtype ilegal (vill=1)")

    @staticmethod
    def _group(reg, emul):
        if emul not in (1, 2, 4, 8) or reg % emul or reg + emul > 32:
            raise IllegalInstruction(f"grupo v{reg} invalido para EMUL={emul}")

    @staticmethod
    def _mask_overlap(vd, vm):
        if not vm and vd == 0:
            raise IllegalInstruction("vd nao pode sobrepor v0 em operacao mascarada")

    # --- instrucoes ------------------------------------------------------------
    def _binop(self, mn, vd, vs2, vs1, vm, fn):
        self._require()
        for r in (vd, vs2, vs1):
            self._group(r, self.lmul)
        self._mask_overlap(vd, vm)
        sew, fmt = self.sew, FMT[self.sew]
        a, b = self.read_vec(vs2, sew), self.read_vec(vs1, sew)
        n = 0
        for i in range(self.vl):
            if self._active(i, vm):
                self.write_elem(vd, sew, i, fn(fmt, a[i], b[i]))
                n += 1
        self.stats.record(mn, n)

    def vtadd(self, vd, vs2, vs1, vm=True):
        self._binop("vtadd.vv", vd, vs2, vs1, vm, lambda f, x, y: f.add(x, y))

    def vtsub(self, vd, vs2, vs1, vm=True):
        self._binop("vtsub.vv", vd, vs2, vs1, vm, lambda f, x, y: f.sub(x, y))

    def vtmul(self, vd, vs2, vs1, vm=True):
        self._binop("vtmul.vv", vd, vs2, vs1, vm, lambda f, x, y: f.mul(x, y))

    def vtdot(self, vd, vs2, vs1, vm=True):
        self._require()
        if 2 * self.sew > 64:
            raise IllegalInstruction("vtdot.vv exige 2*SEW <= 64")
        self._group(vs2, self.lmul)
        self._group(vs1, self.lmul)
        self._group(vd, 1)
        self._mask_overlap(vd, vm)
        sew, wide = self.sew, 2 * self.sew
        xs, ys = self.read_vec(vs2, sew), self.read_vec(vs1, sew)
        act = [self._active(i, vm) for i in range(self.vl)]
        if self.vl == 0:
            self.stats.record("vtdot.vv", 0)
            return
        acc = self.read_elem(vd, wide, 0)
        xs = [x for x, a in zip(xs, act) if a]
        ys = [y for y, a in zip(ys, act) if a]
        self.write_elem(vd, wide, 0, dot_exact(xs, ys, acc, sew, wide))
        self.stats.record("vtdot.vv", len(xs))

    def vncvt(self, vd, vs2, vm=True):
        self._require()
        if 2 * self.sew > 64 or 2 * self.lmul > 8:
            raise IllegalInstruction("vncvt.t.t: origem 2*SEW/2*LMUL fora do limite")
        self._group(vd, self.lmul)
        self._group(vs2, 2 * self.lmul)
        self._mask_overlap(vd, vm)
        sew, wide = self.sew, 2 * self.sew
        src = self.read_vec(vs2, wide)
        n = 0
        for i in range(self.vl):
            if self._active(i, vm):
                u = unpack(src[i], wide)
                if u is None:
                    out = FMT[sew].NAR
                elif u[1] == 0:
                    out = 0
                else:
                    out = encode(u[0], u[1], u[2], sew)
                self.write_elem(vd, sew, i, out)
                n += 1
        self.stats.record("vncvt.t.t", n)

    # --- execucao de palavras / assembly ---------------------------------------
    def execute(self, word: int):
        opcode, funct3 = word & 0x7F, (word >> 12) & 7
        vd, vs1, vs2 = (word >> 7) & 31, (word >> 15) & 31, (word >> 20) & 31
        if opcode == OPC_V and funct3 == F3_VSETVLI and not (word >> 31):
            zimm = (word >> 20) & 0x7FF
            sew = {v: k for k, v in SEW_ENC.items()}.get((zimm >> 3) & 7)
            lmul = {v: k for k, v in LMUL_ENC.items()}.get(zimm & 7)
            rs1, rd = vs1, vd
            if rs1:
                avl = self.x[rs1]
            elif rd:
                avl = 1 << 32                    # AVL = infinito -> VLMAX
            else:
                avl = self.vl                    # mantem vl
            vl = self.vsetvli(avl, sew, lmul, (zimm >> 6) & 1, (zimm >> 7) & 1)
            if rd:
                self.x[rd] = vl
            return
        if opcode == OPC_CUSTOM2 and funct3 == F3_TAKUM and (word >> 26) in BY_FUNCT6:
            mn, kind = BY_FUNCT6[word >> 26]
            vm = bool((word >> 25) & 1)
            if mn == "vtadd.vv":
                self.vtadd(vd, vs2, vs1, vm)
            elif mn == "vtsub.vv":
                self.vtsub(vd, vs2, vs1, vm)
            elif mn == "vtmul.vv":
                self.vtmul(vd, vs2, vs1, vm)
            elif mn == "vtdot.vv":
                self.vtdot(vd, vs2, vs1, vm)
            else:
                self.vncvt(vd, vs2, vm)
            return
        raise IllegalInstruction(f"instrucao ilegal: 0x{word:08X}")

    def run(self, program: str, trace: bool = False):
        for line in program.splitlines():
            w = assemble(line)
            if w is None:
                continue
            if trace:
                print(f"0x{w:08X}  {disassemble(w)}")
            self.execute(w)
