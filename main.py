from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


alpha = 10
beta = 2
K = 150
N = 100
upper_bound = 1
h = 1 / K
tau = upper_bound / N
u = []
x = np.linspace(0.0, 1.0, num=K + 1)
t = []
t_curr = 0
u_curr = [0]*(K + 1)
f = [0]*(K+1)
f_int = []

u.append(u_curr)
u.append([0]*(K + 1))
u.append([0]*(K + 1))

t.append(0)
t.append(t_curr)
t.append(tau)

f_int.append(0)
f_int.append(0)
f_int.append(0)

while t_curr <= upper_bound:
    # print(u)
    tau = upper_bound / N
    u.append([0] * (K + 1))
    while alpha * tau**2 + 4*tau**2/h**2 - 4 > 0:
        print("ouch!")
        tau /= 10
    t_curr += tau
    t.append(t_curr)
    u[-1][0] = (beta*np.sin(t_curr) + (u[-2][1] - u[-2][0])/h)*tau + u[-2][0]
    for i in range(1, K):
        u[-1][i] = 2 * u[-2][i] - u[-3][i] + tau**2 * ( (u[-2][i+1] - 2*u[-2][i] + u[-2][i-1]) / (h**2) - alpha * u[-2][i] / (np.sqrt(1 + u[-2][i]*u[-2][i])))
    u[-1][-1] = u[-2][-1] - tau * (u[-2][-1] - u[-3][-1]) / h
    for i in range(1, K-1):
        u_t_km1 = (u[-1][i-1] - u[-2][i-1])**2/tau**2
        u_t_k = (u[-1][i] - u[-2][i])**2/tau**2
        u_x_km1 = (u[-2][i] - u[-2][i-1])**2/h**2
        u_x_k = (u[-2][i+1] - u[-2][i])**2/h**2
        f[i] = (u_t_k + u_x_k + 2*alpha*(np.sqrt(1 + u[-1][i]) - 1) + u_t_km1 + u_x_km1 + 2*alpha*(np.sqrt(1 + u[-1][i-1]) - 1) )*0.5
        #f[i] = (u_t_k + u_x_k + 2 * alpha * (np.sqrt(1 + u[-2][i]) - 1))
        #f[i] = (u_t_k + u_t_km1) * 0.5
        print(f[i])
    f_int.append(h*np.sum(f))
print(f_int)

t_tofile = []
u_tofile = []
for i in range(0,len(t),40):
    t_tofile.append(t[i])
    u_tofile.append(u[i])

u=pd.DataFrame(data=u,index=t, columns=x)
u_tofile=pd.DataFrame(data=u_tofile,index=t_tofile, columns=x)

f_int = pd.DataFrame(data=f_int,index=t)
u_tofile.to_excel('C:/Users/Лерочка/PycharmProjects/evm4/u10_2.xlsx')
# f_int.to_excel('C:/Прак/4 курс/Python/EVM4/f2.xlsx')
u=np.array(u)


fig = plt.figure(figsize=(20, 20))
ax = Axes3D(fig)
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, t)
Z = u.reshape(Y.shape)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('ось х')
ax.set_ylabel('ось времени')
ax.set_zlabel('ось значений u(x,t)')
plt.show()

fig, ax = plt.subplots()
ax.plot(t,f_int)
ax.grid()
plt.show()