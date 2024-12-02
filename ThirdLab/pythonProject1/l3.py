import numpy as np
from scipy.linalg import lu
from scipy import stats

A = np.array([[4, 3],
              [6, 3]])
P, L, U = lu(A)

print("Преобразованная матрица P:")
print(P)
print("\nМатрица L:")
print(L)
print("\nМатрица U:")
print(U)

det_L = np.linalg.det(L)
det_U = np.linalg.det(U)

det_A = det_L * det_U
print(det_A)

uniform_sample = np.random.randint(0, 100, 100)
normal_sample = np.random.normal(50, 10, 100)
normal_sample = np.clip(normal_sample, 0, 100).astype(int)

print("Выборка с равномерным распределением:", uniform_sample)
print("Выборка с нормальным распределением:", normal_sample)


def compute_statistics(sample):
    mean = np.mean(sample)
    mode = stats.mode(sample)[0]
    modeA = stats.mode(sample)[1]
    median = np.median(sample)
    minimum = np.min(sample)
    maximum = np.max(sample)
    std_dev = np.std(sample)

    return mean, mode, modeA, median, minimum, maximum, std_dev

uniform_stats = compute_statistics(uniform_sample)
normal_stats = compute_statistics(normal_sample)

print("Статистика для выборки с равномерным распределением:")
print(f"Среднее: {uniform_stats[0]}")
print(f"Мода: {uniform_stats[1]} и количество {uniform_stats[2]}")
print(f"Медиана: {uniform_stats[3]}")
print(f"Минимум: {uniform_stats[4]}")
print(f"Максимум: {uniform_stats[5]}")
print(f"Стандартное отклонение: {uniform_stats[6]}")

print("\nСтатистика для выборки с нормальным распределением:")
print(f"Среднее: {normal_stats[0]}")
print(f"Мода: {normal_stats[1]} и количество {normal_stats[2]}")
print(f"Медиана: {normal_stats[3]}")
print(f"Минимум: {normal_stats[4]}")
print(f"Максимум: {normal_stats[5]}")
print(f"Стандартное отклонение: {normal_stats[6]}")

observed_counts = np.bincount(uniform_sample, minlength=100)
expected_counts = np.full(100, len(uniform_sample) / 100)
chi2_statistic, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)


print(f"Статистика хи-квадрат: {chi2_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("Отвергаем нулевую гипотезу: распределение выборки не равномерное.")
else:
    print("Нет оснований для отвержения нулевой гипотезы: распределение выборки равномерное.")

observed_counts = np.bincount(normal_sample, minlength=100)
expected_counts = np.full(100, len(normal_sample) / 100)
chi2_statistic, p_value = stats.chisquare(observed_counts, f_exp=expected_counts)

print(f"Статистика хи-квадрат: {chi2_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("Отвергаем нулевую гипотезу: распределение выборки не равномерное.")
else:
    print("Нет оснований для отвержения нулевой гипотезы: распределение выборки равномерное.")