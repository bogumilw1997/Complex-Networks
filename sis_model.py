import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams["figure.figsize"] = [10, 10]
plt.rcParams['font.size'] = '13'

p0_inf = 0.1

T = 100
beta = 0.12
mu = 0.105

t = np.arange(T)

def przelicz_sis(p0_inf, T, beta, mu):
    
    s_list_model = np.zeros(T)
    i_list_model = np.zeros(T)
    r_list_model = np.zeros(T)

    s_list_model[0] = (1-p0_inf)
    i_list_model[0] = p0_inf

    lambd = beta / mu
    R0 = lambd * s_list_model[0]
    
    # print(f'Tempo empidemii: {lambd}')
    # print(f'{R0 = }')
    
    for i in range(1, T):
        
        s_next = s_list_model[i-1] - beta * s_list_model[i-1] * i_list_model[i-1] + mu * i_list_model[i-1]
        
        if 0 <= s_next <= 1:
            s_list_model[i] = s_next
        elif s_next > 1:
            s_list_model[i] = 1
        else:
            s_list_model[i] = 0
            
        i_next = i_list_model[i-1] + beta * s_list_model[i-1] * i_list_model[i-1] - mu * i_list_model[i-1]
        
        if 0 <= i_next <= 1:
            i_list_model[i] = i_next
        elif i_next > 1:
            i_list_model[i] = 1
        else:
            i_list_model[i] = 0
        
    return R0, lambd, s_list_model, i_list_model

beta = 0.1
mu = 0.15

R0, lambd, s_list_model, i_list_model = przelicz_sis(p0_inf, T, beta, mu)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(t, s_list_model, label = 'S - model', color='blue', linestyle='dashed')
ax1.plot(t, i_list_model, label = 'I - model', color='red', linestyle='dashed')
ax1.set(xlabel='t [krok]', title = 'Względna ilość osób w każdej grupie modelu SIS dla ' + r'$R_0$ = ' + f'{round(R0, 2)}')
ax1.set_yticks(np.linspace(0, 1, 11))
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.legend()

beta = 0.16
mu = 0.1

R0, lambd, s_list_model, i_list_model = przelicz_sis(p0_inf, T, beta, mu)

ax2.plot(t, s_list_model, label = 'S - model', color='blue', linestyle='dashed')
ax2.plot(t, i_list_model, label = 'I - model', color='red', linestyle='dashed')
ax2.set(xlabel='t [krok]', title = 'Względna ilość osób w każdej grupie modelu SIS dla ' + r'$R_0$ = ' + f'{round(R0, 2)}')
ax2.set_yticks(np.linspace(0, 1, 11))
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax2.legend()
plt.show()