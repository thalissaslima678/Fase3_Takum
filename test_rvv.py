"""Testes automaticos da Fase 2.  Execucao:  python -m unittest -v test_rvv"""
import random
import unittest
from fractions import Fraction

from rvv_takum import (FMT, INSTR, IllegalInstruction, VectorMachine, assemble,
                       disassemble, dot_exact)


def vec(rng, sew, n, allow_nar=False):
    out = []
    while len(out) < n:
        b = rng.getrandbits(sew)
        if b != FMT[sew].NAR or allow_nar:
            out.append(b)
    return out


class TestEncoding(unittest.TestCase):
    def test_known_word(self):
        w = assemble("vtadd.vv v1, v2, v3")
        self.assertEqual(w, (1 << 25) | (2 << 20) | (3 << 15) | (1 << 7) | 0x5B)
        self.assertEqual(w & 0x7F, 0b1011011)             # custom-2

    def test_vsetvli_matches_riscv(self):
        self.assertEqual(assemble("vsetvli x5, x10, e16, m1, ta, ma"), 0x0C8572D7)

    def test_roundtrip_all_mnemonics(self):
        rng = random.Random(1)
        for mn, (_, kind) in INSTR.items():
            for _ in range(200):
                vd, vs2, vs1 = (rng.randrange(32) for _ in range(3))
                mask = ", v0.t" if rng.random() < 0.5 else ""
                text = (f"{mn} v{vd}, v{vs2}, v{vs1}{mask}" if kind == "vv"
                        else f"{mn} v{vd}, v{vs2}{mask}")
                self.assertEqual(disassemble(assemble(text)), text)

    def test_distinct_encodings(self):
        words = {assemble(f"{mn} v1, v2" + (", v3" if k == "vv" else ""))
                 for mn, (_, k) in INSTR.items()}
        self.assertEqual(len(words), len(INSTR))

    def test_invalid(self):
        with self.assertRaises(ValueError):
            assemble("vtfoo.vv v1, v2, v3")
        with self.assertRaises(ValueError):
            assemble("vtadd.vv v1, v2")
        with self.assertRaises(IllegalInstruction):
            VectorMachine().execute(0xFFFFFFFF)


class TestConfig(unittest.TestCase):
    def test_vl(self):
        for sew in (8, 16, 32, 64):
            for lmul in (1, 2, 4, 8):
                m = VectorMachine(128)
                vlmax = lmul * 128 // sew
                for avl in (0, 1, vlmax - 1, vlmax, vlmax + 5):
                    self.assertEqual(m.vsetvli(max(avl, 0), sew, lmul), min(max(avl, 0), vlmax))

    def test_vsetvli_instruction_and_scalar_regs(self):
        m = VectorMachine(128)
        m.x[10] = 3
        m.run("vsetvli x5, x10, e32, m2, ta, ma")
        self.assertEqual((m.x[5], m.vl, m.sew, m.lmul), (3, 3, 32, 2))
        m.run("vsetvli x6, x0, e8, m1, ta, ma")           # AVL = VLMAX
        self.assertEqual((m.x[6], m.vl), (16, 16))

    def test_vill(self):
        m = VectorMachine()
        m.vsetvli(4, 128, 1)
        self.assertTrue(m.vill)
        with self.assertRaises(IllegalInstruction):
            m.vtadd(1, 2, 3)


class TestElementwise(unittest.TestCase):
    def test_all_sew_lmul_match_scalar_model(self):
        rng = random.Random(2)
        for sew in (8, 16, 32, 64):
            for lmul in (1, 2, 4, 8):
                m = VectorMachine(128)
                vl = m.vsetvli(10 ** 6, sew, lmul)
                fmt = FMT[sew]
                a, b = vec(rng, sew, vl, True), vec(rng, sew, vl, True)
                m.write_vec(16, sew, a)
                m.write_vec(24, sew, b)
                for op, ref in (("vtadd", fmt.add), ("vtsub", fmt.sub), ("vtmul", fmt.mul)):
                    getattr(m, op)(8, 16, 24)
                    self.assertEqual(m.read_vec(8, sew), [ref(x, y) for x, y in zip(a, b)],
                                     f"{op} e{sew} m{lmul}")

    def test_operand_order_sub(self):
        m = VectorMachine(128)
        m.vsetvli(2, 16, 1)
        f = FMT[16]
        m.write_vec(2, 16, [f.from_float(5.0), f.from_float(1.0)])
        m.write_vec(3, 16, [f.from_float(2.0), f.from_float(4.0)])
        m.run("vtsub.vv v1, v2, v3")                       # vs2 - vs1
        self.assertEqual([f.to_float(v) for v in m.read_vec(1, 16)], [3.0, -3.0])

    def test_mask_and_tail_undisturbed(self):
        m = VectorMachine(128)
        m.vsetvli(5, 16, 1)                                # VLMAX = 8
        f = FMT[16]
        m.write_vec(1, 16, [0x1111] * 8)                   # sentinela
        m.write_vec(2, 16, [f.from_float(1.0)] * 8)
        m.write_vec(3, 16, [f.from_float(2.0)] * 8)
        m.write_mask([1, 0, 1, 0, 1])
        m.run("vtadd.vv v1, v2, v3, v0.t")
        out = m.read_vec(1, 16, 8)
        three = f.from_float(3.0)
        self.assertEqual(out, [three, 0x1111, three, 0x1111, three, 0x1111, 0x1111, 0x1111])

    def test_masked_vd_v0_illegal(self):
        m = VectorMachine()
        m.vsetvli(4, 16, 1)
        with self.assertRaises(IllegalInstruction):
            m.vtadd(0, 2, 3, vm=False)

    def test_group_alignment(self):
        m = VectorMachine()
        m.vsetvli(4, 16, 2)
        with self.assertRaises(IllegalInstruction):
            m.vtadd(1, 2, 4)                               # vd=v1 nao alinhado a LMUL=2
        with self.assertRaises(IllegalInstruction):
            m.vtadd(31, 2, 4)                              # vd=v31: desalinhado e alem de v31


class TestDot(unittest.TestCase):
    def ref(self, sew, acc, xs, ys):
        s = FMT[2 * sew].to_fraction(acc)
        for x, y in zip(xs, ys):
            s += FMT[sew].to_fraction(x) * FMT[sew].to_fraction(y)
        return FMT[2 * sew].from_fraction(s)

    def test_matches_exact_reference(self):
        rng = random.Random(3)
        for sew in (8, 16, 32):
            for _ in range(300):
                m = VectorMachine(128)
                vl = m.vsetvli(10 ** 6, sew, 1)
                xs, ys = vec(rng, sew, vl), vec(rng, sew, vl)
                acc = vec(rng, 2 * sew, 1)[0]
                m.write_vec(2, sew, xs)
                m.write_vec(3, sew, ys)
                m.write_vec(1, 2 * sew, [acc])
                m.run("vtdot.vv v1, v2, v3")
                self.assertEqual(m.read_elem(1, 2 * sew, 0), self.ref(sew, acc, xs, ys))

    def test_single_rounding_beats_chained(self):
        """A soma exata deve ter erro <= a encadeada (mul+add arredondados)."""
        rng = random.Random(4)
        sew, better, worse = 8, 0, 0
        f, w = FMT[8], FMT[16]
        for _ in range(500):
            xs, ys = vec(rng, 8, 16), vec(rng, 8, 16)
            exact = sum(f.to_fraction(x) * f.to_fraction(y) for x, y in zip(xs, ys))
            fused = w.to_fraction(dot_exact(xs, ys, 0, 8, 16))
            chained = 0
            for x, y in zip(xs, ys):
                chained = w.add(chained, w.from_fraction(f.to_fraction(x) * f.to_fraction(y)))
            c = w.to_fraction(chained)
            if abs(fused - exact) < abs(c - exact):
                better += 1
            elif abs(fused - exact) > abs(c - exact):
                worse += 1
        self.assertGreaterEqual(better, worse)

    def test_mask_and_zero_vl(self):
        m = VectorMachine(128)
        f, w = FMT[8], FMT[16]
        m.vsetvli(4, 8, 1)
        m.write_vec(2, 8, [f.from_float(v) for v in (1, 2, 3, 4)])
        m.write_vec(3, 8, [f.from_float(1)] * 4)
        m.write_vec(1, 16, [0])
        m.write_mask([1, 0, 1, 0])
        m.run("vtdot.vv v1, v2, v3, v0.t")
        self.assertEqual(w.to_float(m.read_elem(1, 16, 0)), 4.0)   # 1 + 3
        m.vsetvli(0, 8, 1)
        before = m.read_elem(1, 16, 0)
        m.run("vtdot.vv v1, v2, v3")
        self.assertEqual(m.read_elem(1, 16, 0), before)

    def test_nar_and_limits(self):
        m = VectorMachine(128)
        m.vsetvli(2, 8, 1)
        m.write_vec(2, 8, [FMT[8].NAR, FMT[8].ONE])
        m.write_vec(3, 8, [FMT[8].ONE] * 2)
        m.write_vec(1, 16, [0])
        m.run("vtdot.vv v1, v2, v3")
        self.assertEqual(m.read_elem(1, 16, 0), FMT[16].NAR)
        m.vsetvli(2, 64, 1)
        with self.assertRaises(IllegalInstruction):
            m.vtdot(1, 2, 3)


class TestNarrow(unittest.TestCase):
    def test_t16_to_t8_exhaustive(self):
        m = VectorMachine(512)
        m.vsetvli(10 ** 6, 8, 4)                           # 256 elementos
        vl = m.vl
        w, n = FMT[16], FMT[8]
        out = [None] * 65536
        for base in range(0, 65536, vl):
            chunk = list(range(base, base + vl))
            m.write_vec(8, 16, chunk)
            m.vncvt(16, 8)
            for p, r in zip(chunk, m.read_vec(16, 8)):
                out[p] = r
        for p in range(65536):
            if p == w.NAR:
                self.assertEqual(out[p], n.NAR)
            else:
                self.assertEqual(out[p], n.from_fraction(w.to_fraction(p)), hex(p))
        # monotonia (ordem de complemento de dois)
        order = sorted((p for p in range(65536) if p != w.NAR), key=w.to_signed)
        seq = [n.to_signed(out[p]) for p in order]
        self.assertEqual(seq, sorted(seq))

    def test_exact_values_survive(self):
        m = VectorMachine(128)
        m.vsetvli(16, 8, 1)
        w, n = FMT[16], FMT[8]
        vals = [p for p in range(256) if p != n.NAR]
        for base in range(0, len(vals), 8):
            chunk = vals[base:base + 8]
            m.vsetvli(len(chunk), 8, 1)
            m.write_vec(2, 16, [w.from_fraction(n.to_fraction(p)) for p in chunk])
            m.vncvt(4, 2)
            self.assertEqual(m.read_vec(4, 8), chunk)

    def test_wider_formats_random(self):
        rng = random.Random(5)
        for sew in (16, 32):
            m = VectorMachine(256)
            m.vsetvli(10 ** 6, sew, 1)
            w, n = FMT[2 * sew], FMT[sew]
            for _ in range(200):
                src = vec(rng, 2 * sew, m.vl, True)
                m.write_vec(2, 2 * sew, src)
                m.vncvt(8, 2)
                for p, r in zip(src, m.read_vec(8, sew)):
                    exp = n.NAR if p == w.NAR else n.from_fraction(w.to_fraction(p))
                    self.assertEqual(r, exp)

    def test_saturation_and_mask(self):
        m = VectorMachine(128)
        m.vsetvli(2, 8, 1)
        w, n = FMT[16], FMT[8]
        m.write_vec(2, 16, [w.MAXPOS, w.MINPOS])
        m.write_vec(4, 8, [0x55, 0x55])
        m.vncvt(4, 2)
        self.assertEqual(m.read_vec(4, 8), [n.MAXPOS, n.MINPOS])   # sem NaR / zero
        m.write_vec(4, 8, [0x55, 0x55])
        m.write_mask([0, 1])
        m.run("vncvt.t.t v4, v2, v0.t")
        self.assertEqual(m.read_vec(4, 8), [0x55, n.MINPOS])

    def test_illegal(self):
        m = VectorMachine(128)
        m.vsetvli(2, 64, 1)
        with self.assertRaises(IllegalInstruction):
            m.vncvt(4, 2)
        m.vsetvli(2, 8, 8)
        with self.assertRaises(IllegalInstruction):
            m.vncvt(8, 16)                                  # EMUL da origem = 16
        m.vsetvli(2, 8, 2)
        with self.assertRaises(IllegalInstruction):
            m.vncvt(4, 2)                                   # origem exige alinhamento a 4


class TestStats(unittest.TestCase):
    def test_counters(self):
        m = VectorMachine(128)
        m.x[10] = 8
        m.run("""
            vsetvli x5, x10, e16, m1, ta, ma
            vtadd.vv v1, v2, v3
            vtmul.vv v1, v2, v3
            vtadd.vv v1, v2, v3
        """)
        self.assertEqual(m.stats.instructions, 4)
        self.assertEqual(m.stats.by_mnemonic["vtadd.vv"], 2)
        self.assertEqual(m.stats.element_ops, 24)


if __name__ == "__main__":
    unittest.main()
