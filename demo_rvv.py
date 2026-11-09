"""Demonstracao da Fase 2:  python demo_rvv.py"""
from fractions import Fraction

from rvv_takum import FMT, VectorMachine, assemble, disassemble


def demo_asm():
    print("=== 1) Assembly Zvtakum + codificacao binaria ===")
    m = VectorMachine(vlen=128)
    f = FMT[16]
    m.x[10] = 8
    m.write_vec(2, 16, [f.from_float(v) for v in (1, 2, 3, 4, 5, 6, 7, 8)])
    m.write_vec(3, 16, [f.from_float(v) for v in (0.5,) * 8])
    m.run("""
        vsetvli x5, x10, e16, m1, ta, ma
        vtadd.vv v1, v2, v3
        vtmul.vv v4, v2, v3
        vtsub.vv v5, v2, v3
    """, trace=True)
    for r in (1, 4, 5):
        print(f"v{r} =", [f.to_float(v) for v in m.read_vec(r, 16)])


def demo_stripmine():
    print("\n=== 2) Strip-mining: soma de 20 elementos com VLEN=128, e16 ===")
    m = VectorMachine(vlen=128)
    f = FMT[16]
    a = [f.from_float(i) for i in range(20)]
    b = [f.from_float(100 - i) for i in range(20)]
    out, i, passes = [], 0, 0
    while i < 20:
        m.x[10] = 20 - i
        m.run("vsetvli x5, x10, e16, m1, ta, ma")
        vl = m.x[5]
        m.write_vec(2, 16, a[i:i + vl]); m.write_vec(3, 16, b[i:i + vl])
        m.run("vtadd.vv v1, v2, v3")
        out += m.read_vec(1, 16); i += vl; passes += 1
    print(f"{passes} iteracoes, resultado ok:",
          all(f.to_float(o) == 100.0 for o in out))
    print("instrucoes executadas:", m.stats.instructions, m.stats.by_mnemonic)


def demo_dot():
    print("\n=== 3) vtdot.vv: soma exata (T8 -> acumulador T16) ===")
    f8, f16 = FMT[8], FMT[16]
    m = VectorMachine(vlen=128)
    m.vsetvli(16, 8, 1)
    # cancelamento: 64 + 14 parcelas pequenas - 64  (caso ruim p/ soma encadeada)
    xs = [f8.from_float(64.0)] + [f8.from_float(0.09)] * 14 + [f8.from_float(-64.0)]
    ys = [f8.ONE] * 16
    m.write_vec(2, 8, xs); m.write_vec(3, 8, ys); m.write_vec(1, 16, [0])
    m.run("vtdot.vv v1, v2, v3")
    exact = sum(f8.to_fraction(x) * f8.to_fraction(y) for x, y in zip(xs, ys))
    fused = f16.to_fraction(m.read_elem(1, 16, 0))
    chained = 0                                   # mul e add arredondados em T16
    for x, y in zip(xs, ys):
        chained = f16.add(chained, f16.from_fraction(f8.to_fraction(x) * f8.to_fraction(y)))
    c = f16.to_fraction(chained)
    print(f"exato     = {float(exact):.8f}")
    print(f"vtdot     = {float(fused):.8f}  (erro {float(abs(fused - exact)):.2e})")
    print(f"encadeado = {float(c):.8f}  (erro {float(abs(c - exact)):.2e})")


def demo_ncvt():
    print("\n=== 4) vncvt.t.t: T16 -> T8 ===")
    m = VectorMachine(vlen=128)
    w, n = FMT[16], FMT[8]
    vals = [0.1, 1.0, 3.14159, -100.0, 1e20, 1e-20, 0.0, float("nan")]
    m.vsetvli(8, 8, 1)
    m.write_vec(2, 16, [w.from_float(v) for v in vals])
    m.run("vncvt.t.t v4, v2")
    for v, s, d in zip(vals, m.read_vec(2, 16), m.read_vec(4, 8)):
        print(f"x={v:<10g} T16=0x{s:04X} -> T8=0x{d:02X} ({n.to_float(d):.6g})")


if __name__ == "__main__":
    demo_asm(); demo_stripmine(); demo_dot(); demo_ncvt()
