# Takum-RVV — Fase 1: Decodificação e Aritmética Takum

Projeto Integrador AOC 2026/02 (DELT/UFMG). Modelo funcional em Python do formato
**linear takum** (Hunhold, 2024) para T8, T16 e T32.

## Estrutura
| Arquivo | Conteúdo |
|---|---|
| `rvv_takum.py` | **Fase 2:** simulador RVV + Zvtakum (instruções, assembler, CSRs, máscara, LMUL) |
| `test_rvv.py` | **Fase 2:** 23 testes (codificação, `vl`, elemento a elemento, máscara, `vtdot`, `vncvt`) |
| `demo_rvv.py` | **Fase 2:** demonstração em assembly, strip-mining, `vtdot` e `vncvt` |
| `takum.py` | decodificador/codificador (S, D, R, C, F), `add`, `sub`, `mul`, `neg` |
| `test_takum.py` | testes automáticos (exaustivos em T8/T16, amostrais em T32) |
| `demo.py` | demonstração de campos, operações e alcance dinâmico |

## Uso
```python
from takum import T16
a, b = T16.from_float(0.1), T16.from_float(0.2)
print(T16.fields(a).pretty())        # S, D, R, C, F
print(T16.to_float(T16.add(a, b)))
```
Testes: `python -m unittest -v test_takum`  ·  Demo: `python demo.py`

## Formato
`| S | D | R(3) | C(r) | F(p=n-5-r) |`, com `r = R` se `D=1`, `r = 7-R` se `D=0`;
`c = 2^r-1+C` (D=1) ou `c = -2^(r+1)+1+C` (D=0); valor `(1+f)·2^c` (S=0) ou
`(-2+f)·2^-(c+1)` (S=1). `0…0` = zero, `10…0` = NaR. Negativos = complemento de dois,
logo a ordem dos valores coincide com a ordem dos padrões como inteiros com sinal.

## Decisões de projeto
- **Decodificação comum:** padrões com n < 12 são estendidos com zeros até 12 bits.
  Consequência: no T8 só há 3 bits após S/D/R, então a característica é truncada
  (alcance T8: c ∈ [-239, 239]; T16/T32: c ∈ [-255, 254]).
- **Aritmética exata + um único arredondamento:** operandos viram `(sinal, M, e)`;
  soma e produto são feitos em inteiros exatos e arredondados uma vez no codificador
  (round-to-nearest-even sobre a cadeia de bits).
- **Saturação:** nunca há underflow para 0 nem overflow para NaR.
- **NaR:** propaga em qualquer operação.

---

# Fase 2 — Extensão Zvtakum (RVV)

Execução: `python -m unittest -v test_rvv` · `python demo_rvv.py`

## Instruções
| Instrução | Semântica (SEW seleciona o formato: e8→T8, e16→T16, e32→T32) |
|---|---|
| `vtadd.vv vd, vs2, vs1[, v0.t]` | `vd[i] = vs2[i] + vs1[i]` |
| `vtsub.vv vd, vs2, vs1[, v0.t]` | `vd[i] = vs2[i] - vs1[i]` |
| `vtmul.vv vd, vs2, vs1[, v0.t]` | `vd[i] = vs2[i] * vs1[i]` |
| `vtdot.vv vd, vs2, vs1[, v0.t]` | `vd[0] (2·SEW) += Σ vs2[i]·vs1[i]`, soma exata, **um único arredondamento** |
| `vncvt.t.t vd, vs2[, v0.t]` | `vd[i] (SEW) = arred(vs2[i] (2·SEW))` |

## Codificação proposta
Opcode **custom-2** (`0x5B`, reservado pela especificação RISC-V a extensões de fabricante, logo
sem conflito com a RVV padrão), layout do OP-V:
`| funct6 | vm | vs2 | vs1 | 000 | vd | 1011011 |`.
`funct6`: `vtadd`=000000, `vtsub`=000010, `vtmul`=100100, `vtdot`=111001, `vncvt`=010010 (`vs1`=0).
Os valores de `funct6` são arbitrários dentro do espaço custom.

## Decisões de projeto
- **Sem novos CSRs e sem `frm`/`fflags`:** Takum tem um único modo de arredondamento e nenhuma flag
  de exceção; só os CSRs vetoriais padrão (`vl`, `vtype`, `vlenb`) são usados.
- **Um conjunto de instruções para todas as larguras:** o formato vem do SEW (`vsetvli`), não do mnemônico.
- **`vtdot.vv`:** destino com EEW = 2·SEW (T8→T16, T16→T32, T32→T64), reduzido ao elemento 0;
  os produtos e o acumulador anterior são somados em aritmética exata (estilo *quire*).
- **`vncvt.t.t`:** origem com EEW = 2·SEW; arredondamento único, com saturação (sem zero/NaR espúrios).
- **Máscara e cauda:** política *undisturbed* (permitida pela política *agnostic*); `vd` não pode sobrepor `v0` quando mascarado.
- **Cargas/stores:** são independentes do formato (`vle*.v`/`vse*.v` apenas movem bits), não exigem instruções novas.
- **Fora do escopo do modelo:** LMUL fracionário (gera `vill`), regras de sobreposição de grupos em narrowing, `vstart`/exceções precisas.
- **Contadores:** `VectorMachine.stats` registra instruções e operações por elemento (insumo para a Fase 3).
