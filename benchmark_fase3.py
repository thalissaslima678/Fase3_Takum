import random
from fractions import Fraction
from rvv_takum import FMT, VectorMachine

def gerar_matriz_esparsa(linhas, colunas, densidade=0.1, seed=42):
    random.seed(seed)
    A = [[0.0] * colunas for _ in range(linhas)]
    for i in range(linhas):
        for j in range(colunas):
            if random.random() < densidade:
                A[i][j] = random.uniform(-10, 10)
    return A

def matvec_takum(A, x_float, n_bits=16):
    m = VectorMachine(vlen=128)
    fmt = FMT[n_bits]
    linhas = len(A)
    colunas = len(x_float)
    
    m.vsetvli(10**6, n_bits, 1) 
    vl_max = m.vl
    
    y = []
    for i in range(linhas):
        linha = A[i]
        acc_bits = 0 
        
        for start in range(0, colunas, vl_max):
            fim = min(start + vl_max, colunas)
            vl = fim - start
            a_bits = [fmt.from_float(v) for v in linha[start:fim]]
            x_bits = [fmt.from_float(v) for v in x_float[start:fim]]
            
            m.vsetvli(vl, n_bits, 1)
            m.write_vec(2, n_bits, a_bits)
            m.write_vec(3, n_bits, x_bits)
            m.write_elem(1, 2*n_bits, 0, acc_bits)
            
            m.run("vtdot.vv v1, v2, v3")
            acc_bits = m.read_elem(1, 2*n_bits, 0)
            
        y.append(FMT[2*n_bits].to_float(acc_bits))
        
    return y, m.stats

def main():
    print("=== BENCHMARK FASE 3 ===")
    linhas, colunas, densidade = 32, 64, 0.1
    A = gerar_matriz_esparsa(linhas, colunas, densidade)
    x_float = [random.uniform(-5, 5) for _ in range(colunas)]
    
    y_ref = [sum(A[i][j] * x_float[j] for j in range(colunas)) for i in range(linhas)]
    
    for bits in (8, 16, 32):
        y_takum, stats = matvec_takum(A, x_float, n_bits=bits)
        erros = []
        for r, t in zip(y_ref, y_takum):
            if abs(r) > 1e-12:
                erros.append(abs(t - r) / abs(r))
            else:
                erros.append(abs(t - r))
                
        erro_medio = sum(erros) / len(erros)
        erro_max = max(erros)
        
        print(f"\n--- Formato T{bits} ---")
        print(f"Instrucoes executadas: {stats.instructions}")
        print(f"Erro relativo medio: {erro_medio:.4e}")
        print(f"Erro relativo maximo: {erro_max:.4e}")

if __name__ == "__main__":
    main()