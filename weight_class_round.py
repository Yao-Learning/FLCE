import matplotlib.pyplot as plt
import numpy as np


def norm_list(list, round_num=4):
    list_norm = list / sum(list)
    list_norm_rounded = [round(num, round_num) for num in list_norm]
    list_norm_rounded[-1] = round(1 - sum(list_norm_rounded[:-1]), round_num)
    return list_norm_rounded


x = np.arange(0, 2, 0.2)

y = np.exp(x)

y = np.arange(1, 11, 1)



y_norm_rounded = norm_list(y)

# test_value = sum(y_norm_rounded)
# print(f"-----种类权重分配方案1-----")
# print(f"test_value: {test_value}")
# print(f"y_norm_rounded: {y_norm_rounded}")
# print(f"max(y_norm_rounded: {max(y_norm_rounded)}")
# print(f"min(y_norm_rounded: {min(y_norm_rounded)}")
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(y_norm_rounded, '-o')  # 使用线和点来表示数据点
# plt.title('Norm(1:11:1)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.grid(True)
# # plt.show()
#
#
#
# y = np.arange(10, 0, -1)
#
# y_norm_rounded = norm_list(y)
#
# test_value = sum(y_norm_rounded)
# print(f"-----种类权重分配方案2-----")
# print(f"test_value: {test_value}")
# print(f"y_norm_rounded: {y_norm_rounded}")
# print(f"max(y_norm_rounded: {max(y_norm_rounded)}")
# print(f"min(y_norm_rounded: {min(y_norm_rounded)}")
# # 绘制图形
# plt.figure(figsize=(8, 6))
# plt.plot(y_norm_rounded, '-o')  # 使用线和点来表示数据点
# plt.title('Norm(10:0:-1)')
# plt.xlabel('Classes')
# plt.ylabel('Class Weight')
# plt.grid(True)


# plt.show()

x = np.arange(1, 1001, 1)
y = np.log(x)

y_norm_rounded = norm_list(y, round_num=8)

test_value = sum(y_norm_rounded)
print(f"-----轮次权重分配方案1-----")
print(f"test_value: {test_value}")
print(f"y_norm_rounded: {y_norm_rounded}")
print(f"max(y_norm_rounded: {max(y_norm_rounded)}")
print(f"min(y_norm_rounded: {min(y_norm_rounded)}")
# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(y_norm_rounded, '-', linewidth=4)  # 使用线和点来表示数据点
# plt.title('Norm(log(1:1000:1))')
plt.xlabel('Rounds', fontsize=20)
plt.ylabel('Round Weight', fontsize=20)
plt.grid(True)
# plt.tight_layout()
plt.legend()
plt.title('(b) Increase scheme', fontsize=24)
plt.savefig('/data/yaominghao/code/FedRepo/result/picture/FLCE_Round_Weight2.pdf', bbox_inches='tight')
plt.show()


x = np.arange(1, 11, 0.01)
y = 1 / x

y_norm_rounded = norm_list(y, round_num=8)

test_value = sum(y_norm_rounded)
print(f"-----轮次权重分配方案2-----")
print(f"test_value: {test_value}")
print(f"y_norm_rounded: {y_norm_rounded}")
print(f"max(y_norm_rounded: {max(y_norm_rounded)}")
print(f"min(y_norm_rounded: {min(y_norm_rounded)}")
# 绘制图形
plt.figure(figsize=(10, 6))
plt.plot(y_norm_rounded, '-', linewidth=4)  # 使用线和点来表示数据点
# plt.title('Norm(1 / (1:11.01:0.01))')

plt.xlabel('Rounds', fontsize=20)
plt.ylabel('Round Weight', fontsize=20)
plt.grid(True)
# plt.tight_layout()


plt.title('(a) Decrease scheme', fontsize=24)
plt.savefig('/data/yaominghao/code/FedRepo/result/picture/FLCE_Round_Weight.pdf', bbox_inches='tight')
plt.show()