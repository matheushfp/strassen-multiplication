import numpy as np
import timeit


# Gera duas matrizes aleatórias n x n (float)
def random_matrices_float(n):
    m1 = np.random.rand(n, n)
    m2 = np.random.rand(n, n)
    return m1, m2


# Gera duas matrizes aleatórias n x n (int)
def random_matrices(n):
    m1 = np.random.randint(11, size=(n, n))
    m2 = np.random.randint(11, size=(n, n))
    return m1, m2


# Cria uma matriz e preenche com 0
def new_matrix(r, c):
    matrix = np.zeros((r, c))
    return matrix


def common_multiply(m1, m2):
    # Verifica se são matrizes n x m e m x p
    if len(m1[0]) != len(m2):
        return 'Não é possível realizar a multiplicação'
    else:
        # Inicializa a matriz que armazenará a multiplicação com 0s
        m3 = new_matrix(len(m1), len(m2[0]))
        # Laço que efetua a multiplicação das matrizes
        for i in range(len(m1)):
            for j in range(len(m2[0])):
                for k in range(len(m2)):
                    m3[i][j] = m3[i][j] + m1[i][k] * m2[k][j]
        return m3


# Divide uma matriz n x m
def split(m1):
    n, m = m1.shape
    
    n = n//2
    m = m//2
    
    a = m1[:n, :m]
    b = m1[:n, m:]
    c = m1[n:, :m]
    d = m1[n:, m:]
    
    return a, b, c, d


def strassen_multiply(m1, m2):
    n = len(m1)
    
    # Melhor cenário: Matrizes com apenas 1 elemento
    if n == 1:
        return m1 * m2
    else:
        # Divide as duas matrizes
        a, b, c, d = split(m1)
        e, f, g, h = split(m2)
        
        # p1 = a(f - h)
        p1 = strassen_multiply(a, np.subtract(f, h))
        
        # p2 = (a + b)h
        p2 = strassen_multiply(np.add(a, b), h)
        
        # p3 = (c + d)e
        p3 = strassen_multiply(np.add(c, d), e)
        
        # p4 = d(g - e)
        p4 = strassen_multiply(d, np.subtract(g, e)) 
        
        # p5 = (a + d)(e + h)
        p5 = strassen_multiply(np.add(a, d), np.add(e, h))
        
        # p6 = (b - d)(g + h)
        p6 = strassen_multiply(np.subtract(b, d), np.add(g, h))
        
        # p7 = (a - c)(e + f)
        p7 = strassen_multiply(np.subtract(a, c), np.add(e, f))

        m3_a = np.add(np.subtract(np.add(p5, p4), p2), p6)
        m3_b = np.add(p1, p2)  
        m3_c = np.add(p3, p4)
        m3_d = np.subtract(np.subtract(np.add(p1, p5), p3),  p7)
        
        # Une as submatrizes e transforma na matriz final
        # Concatena as matrizes da metade esquerda, depois da metade direita e as une
        left_half = np.vstack((m3_a, m3_c))
        right_half = np.vstack((m3_b, m3_d))
        
        m3 = np.hstack((left_half, right_half))
        
        return m3


# Execução
n = 4
m1, m2 = random_matrices(n)

# Cálculo do Tempo para a execução (Common)
inicio = timeit.default_timer()
result1 = common_multiply(m1, m2)
fim = timeit.default_timer()
time_common = fim - inicio

# Cálculo do Tempo para a execução (Strassen)
inicio = timeit.default_timer()
result2 = strassen_multiply(m1, m2)
fim = timeit.default_timer()
time_strassen = fim - inicio

print(f'Matriz 1:\n{m1}')
print(f'Matriz 2:\n{m2}')
print(f'Common Multiply:\n{result1}')
print(f'Time (Common Multiply): {time_common}\n')
print(f'Strassen Multiply:\n{result2}')
print(f'Time (Strassen Multiply): {time_strassen}')
