# import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import style
# plt.figure(figsize=(10, 5), dpi=70)  # 设置图像大小
#
# x_labels = ['IID', 'IID_Noise', 'Non-IID', 'Non-IID_Noise']
# x_values = [
#     [2.23, 2.24, 5.01, 5.01],
#     [0.06, 0.13, 0.35, 1.26],
# ]
# models = ['Fedavg', 'FedBS']
# # color = ['#4473c5', '#ec7e32', '#a5a5a5']
# color = ['#4473c5', '#ec7e32']
# hatch = ['', '', '']
#
# def draw_time(models, x_labels, x_values, color, hatch):
#     plt.cla()
#     x = np.arange(len(x_labels)) * 2
#     for i in range(2):
#         plt.bar(x + 0.5 * i, x_values[i], color=color[i], hatch=hatch[i], width=0.5, label=models[i], edgecolor='black', linewidth=0.2, alpha=1)
#     plt.xticks(x + 0.60, x_labels)
#
#     ax = plt.gca()
#     # ax.set_yscale('log', basey=10)
#     ax.set_yscale('log')
#     ## ax.set_yticks([10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2)])
#
#     plt.legend(loc="best", prop={"size": 14})
#     plt.xlabel('Data sets', fontsize=16)
#     plt.ylabel('KL divergence', fontsize=16)
#
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#
#     plt.rc('axes', axisbelow=True)
#     plt.grid(linestyle = '--', linewidth = 0.3, color= 'gray', alpha = 0.2)
#     # plt.savefig('time.pdf', dpi=700)
#     plt.show()
#
# draw_time(models, x_labels, x_values, color, hatch)



import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Times New Roman'

name_list = ['IID', 'IID_Noise', 'Non-IID', 'Non-IID_Noise']
num_list = [2.23, 2.24, 5.01, 5.01]
num_list1 = [0.15, 5.97, 0.14, 5.73]
num_list2 = [0.06, 0.13, 0.35, 1.26]
num_list3 = [0.06, 0, 0, 0]

x = list(range(len(num_list)))
total_width, n = 0.8, 3
width = total_width / n

p1 = plt.bar(x, num_list, width=width, label='FedAvg', color='#4473c5')
plt.bar_label(p1, label_type='edge')
for i in range(len(x)):
    x[i] = x[i] + width
p2 = plt.bar(x, num_list1, width=width, label='FedFa', tick_label=name_list, color='#a5a5a5')
plt.bar_label(p2, label_type='edge')
for i in range(len(x)):
    x[i] = x[i] + width
p3 = plt.bar(x, num_list2, width=width, label='FLBS', tick_label=name_list, color='#ec7e32')
for i in range(len(x)):
    x[i] = x[i] + width
p4 = plt.bar(x, num_list3, width=width, label='Fedproavg', tick_label=name_list, color='gray')
plt.bar_label(p3, label_type='edge')
plt.xlabel('Data sets', fontsize=16)
plt.ylabel('KL divergence', fontsize=16)
plt.legend()
plt.show()

# plt.savefig('./result/picture/kl.pdf')