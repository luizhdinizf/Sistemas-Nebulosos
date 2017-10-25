import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def erro(y_real,y_simu):
    y2 = np.subtract(y_real,y_simu)
    y2 = np.square(y2)
    y2=np.sqrt(y2)
    rms=np.sum(y2)
    rms=rms/len(y2)
    return rms


def t_norma_min(x1,x2):
    return np.fmin(x1,x2)
def s_norma_max(x1,x2):
    return np.fmax(x1,x2)
def s_norma_sum(x1,x2):
    c = np.add(x1,x2)
    d = np.multiply(x1,x2)  
    return np.subtract(c,d)


def agreggation(x1,x2):
    return s_norma_sum(x1,x2)
def s_norma(x1,x2):
    return s_norma_max(x1,x2)
def t_norma(x1,x2):
    return t_norma_min(x1,x2)
def pertinencia(x,pontos):
    return  fuzz.trimf(x,pontos)
def pertinencia_gauss(x,pontos):
    centro = pontos[1]
    sigma = pontos[2]-pontos[1]
    return fuzz.membership.gaussmf(x,centro,sigma)


x = np.arange(-4, 4, 0.01)
y_real = np.square(x)

y = np.arange(0, 17, 1)


x_proximo_4 = pertinencia(x, [3, 4, 5])
x_proximo_3 = pertinencia(x, [2, 3, 4])
x_proximo_2 = pertinencia(x, [1, 2, 3])
x_proximo_1 = pertinencia(x, [0, 1, 2])
x_proximo_0 = pertinencia(x, [-1, 0, 1])
x_proximo_menos_1 = pertinencia(x, [-2, -1, 0])
x_proximo_menos_2 = pertinencia(x, [-3, -2, -1])
x_proximo_menos_3 = pertinencia(x, [-4, -3, -2])
x_proximo_menos_4 = pertinencia(x, [-5, -4, -3])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))
ax0.grid(color='000000', linestyle='dashed', linewidth=0.4)
ax1.grid(color='000000', linestyle='dashed', linewidth=0.4)
ax2.grid(color='000000', linestyle='dashed', linewidth=0.4)

ax0.plot(x, x_proximo_4, 'r', linewidth=1.5, label='4')
ax0.plot(x, x_proximo_3, 'g', linewidth=1.5, label='3')
ax0.plot(x, x_proximo_2, 'b', linewidth=1.5, label='2')
ax0.plot(x, x_proximo_1, 'r', linewidth=1.5, label='1')
ax0.plot(x, x_proximo_0, 'g', linewidth=1.5, label='0')
ax0.plot(x, x_proximo_menos_1, 'b', linewidth=1.5, label='4')
ax0.plot(x, x_proximo_menos_2, 'r', linewidth=1.5, label='3')
ax0.plot(x, x_proximo_menos_3, 'g', linewidth=1.5, label='2')
ax0.plot(x, x_proximo_menos_4, 'b', linewidth=1.5, label='1')
ax0.set_title('Entradas')
ax0.legend()

y_proximo_16 = pertinencia(y, [9, 16, 20])
y_proximo_9 = pertinencia(y, [4, 9, 16])
y_proximo_4 = pertinencia(y, [1, 4, 9])
y_proximo_1 = pertinencia(y, [0, 1, 4])
y_proximo_0 = pertinencia(y, [0, 0, 1])

ax1.set_title('Saidas')
ax1.legend()
ax1.plot(y, y_proximo_16, 'r', linewidth=1.5, label='16')
ax1.plot(y, y_proximo_9, 'g', linewidth=1.5, label='9')
ax1.plot(y, y_proximo_4, 'b', linewidth=1.5, label='4')
ax1.plot(y, y_proximo_1, 'r', linewidth=1.5, label='1')
ax1.plot(y, y_proximo_0, 'g', linewidth=1.5, label='0')





active_rule0 = x_proximo_0
active_rule1 = s_norma(x_proximo_1, x_proximo_menos_1)
active_rule2 = s_norma(x_proximo_2, x_proximo_menos_2)
active_rule3 = s_norma(x_proximo_3, x_proximo_menos_3)
active_rule4 = s_norma(x_proximo_4, x_proximo_menos_4)

ax2.set_title('Regras')
ax2.legend()
ax2.plot(x, active_rule1, 'r', linewidth=1.5, label='16')
ax2.plot(x, active_rule2, 'g', linewidth=1.5, label='16')
ax2.plot(x, active_rule3, 'b', linewidth=1.5, label='16')
ax2.plot(x, active_rule4, 'r', linewidth=1.5, label='16')


x_simu = np.arange(-4, 4, 0.01)
y_simu = []

fig, ax0 = plt.subplots(nrows=1, figsize=(8, 9))
ax0.grid(color='000000', linestyle='dashed', linewidth=0.4)
for i in x_simu:
    x_test = i
    
    x_proximo_novo_4 = fuzz.interp_membership(x, active_rule4,x_test)
    x_proximo_novo_3 = fuzz.interp_membership(x, active_rule3,x_test)
    x_proximo_novo_2 = fuzz.interp_membership(x, active_rule2,x_test)
    x_proximo_novo_1 = fuzz.interp_membership(x, active_rule1,x_test)
    x_proximo_novo_0 = fuzz.interp_membership(x, active_rule0,x_test)
    
    y_proximo_16_novo =  t_norma(x_proximo_novo_4, y_proximo_16)
    y_proximo_9_novo =  t_norma(x_proximo_novo_3, y_proximo_9)
    y_proximo_4_novo =  t_norma(x_proximo_novo_2, y_proximo_4)
    y_proximo_1_novo =  t_norma(x_proximo_novo_1, y_proximo_1)
    y_proximo_0_novo =  t_norma(x_proximo_novo_0, y_proximo_0)
    
    
    
    saida_fuzzy = agreggation(y_proximo_16_novo,y_proximo_9_novo)
    saida_fuzzy = agreggation(saida_fuzzy,y_proximo_4_novo)
    saida_fuzzy = agreggation(saida_fuzzy,y_proximo_1_novo)
    saida_fuzzy = agreggation(saida_fuzzy,y_proximo_0_novo)
    
    tip = fuzz.defuzz(y, saida_fuzzy, 'centroid')
    print(tip)
    y_simu.append(tip)    
#    ax1.plot(y, saida_fuzzy, 'g', linewidth=1.5, label='16')

rms = erro(y_real,y_simu)

ax0.plot(x, y_real, 'b', linewidth=1.5, label='16')
ax0.plot(x_simu, y_simu, 'r', linewidth=1.5, label='16')