import numpy as np
from scipy.linalg import lu
from scipy import stats

def pretty_print_matrixP(matrix):
    for row in matrix:
        print(" ".join(f"{int(x) if x.is_integer() else x:.2f}" for x in row))

## задание 1

A = np.array([[2, -5, 1, 0],
            [1, -1, -13, 0],
            [3, -2, -2, -4],
            [4, 0, 2.7, -1.3]])

## задание 2

P, L, U = lu(A)

print("Матрица P:") # перестановочная матрица
pretty_print_matrixP(P)

print("\nМатрица L:") # нижняя треугольная матрица
pretty_print_matrixP(L)

print("\nМатрица U:") # верхняя треугольна
pretty_print_matrixP(U)

#pretty_print_matrixP(P @ L @ U) # проверка

## задание 3

detP = np.linalg.det(P)
detPinv = 1 / detP
detU = np.prod(np.diag(U))
detL = np.prod(np.diag(L))

detA = detL * detU * detPinv

print("\nОпределитель:")
print(detA)

## задание 4

RavS = np.random.randint(0, 100, 100).astype(int)

NorS = np.random.normal(50, 10, 100)
NorS = np.clip(NorS, 0, 100).astype(int)

print("\nВыборка с равномерным распределением:", RavS)
print("\nВыборка с нормальным распределением:", NorS)

## задание 5

def compute_statistics(sample):
    mean = np.mean(sample)
    mode = stats.mode(sample)[0]
    modeA = stats.mode(sample)[1]
    median = np.median(sample)
    minimum = np.min(sample)
    maximum = np.max(sample)
    std_dev = np.std(sample)

    return mean, mode, modeA, median, minimum, maximum, std_dev

RavStats = compute_statistics(RavS)
NorStats = compute_statistics(NorS)

print("\nСтатистика для выборки с равномерным распределением:")
print(f"Среднее: {RavStats[0]}")
print(f"Мода: {RavStats[1]} количество {RavStats[2]}")
print(f"Медиана: {RavStats[3]}")
print(f"Минимум: {RavStats[4]}")
print(f"Максимум: {RavStats[5]}")
print(f"Стандартное отклонение: {RavStats[6]}")

print("\nСтатистика для выборки с нормальным распределением:")
print(f"Среднее: {NorStats[0]}")
print(f"Мода: {NorStats[1]} количество {NorStats[2]}")
print(f"Медиана: {NorStats[3]}")
print(f"Минимум: {NorStats[4]}")
print(f"Максимум: {NorStats[5]}")
print(f"Стандартное отклонение: {NorStats[6]}")

##задание 6

print("\n Равномерное распределение")

num_bins = 10 #количество интервалов

oc, bin_edges = np.histogram(RavS, bins=num_bins, range=(0, 100)) # разделение выборки на интервалы
ec = np.full(num_bins, len(RavS) / num_bins) #ожидаемые частоты

chi2_statistic, p_value = stats.chisquare(oc, ec) # вычисление p-value с помощью критерия хи-квадрат


print(f"\nh-квадрат: {chi2_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("\nраспределение неравномерное")
else:
    print("\nраспределение равномерное")


print("\n Нормальное распределение")

oc, bin_edges = np.histogram(NorS, bins=num_bins, range=(0, 100)) # разделение выборки на интервалы
ec = np.full(num_bins, len(NorS) / num_bins) #ожидаемые частоты

chi2_statistic, p_value = stats.chisquare(oc, ec) # вычисление p-value с помощью критерия хи-квадрат


print(f"\nh-квадрат: {chi2_statistic}")
print(f"p-value: {p_value}")

if p_value < 0.05:
    print("\nраспределение неравномерное")
else:
    print("\nраспределение равномерное")
