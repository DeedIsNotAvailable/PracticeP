import numpy
import random as r

numpy.set_printoptions(suppress=False, linewidth=numpy.inf)

def pretty_print_matrix(matrix):
    matrix_str = numpy.array_str(matrix, max_line_width=numpy.inf, suppress_small=True)
    cleaned_str = matrix_str.replace('[', '').replace(']', '').replace('.', '')
    print(' '+cleaned_str)

def pretty_print_matrixP(matrix):
    matrix_str = numpy.array_str(matrix, max_line_width=numpy.inf, suppress_small=True)
    cleaned_str = matrix_str.replace('[', '').replace(']', '')
    print(' '+cleaned_str)

array = numpy.arange(10, 70, 2)

print("\n1) Множество: ")
pretty_print_matrix(array)

matrixA = array.reshape((6, 5))
#print("\n2.1) Матрица А: ")
#pretty_print_matrix(matrixA)

matrixAt = matrixA.transpose()
#print("\n2.2) Матрица А транспонированная: ")
#pretty_print_matrix(matrixAt)

matrixAt = numpy.multiply(matrixAt, 2.5)
#print("\n3.1) Матрица А умноженная на 2.5: ")
#pretty_print_matrix(matrixAt)

matrixAt[0] = matrixAt[0] - 5
print("\n3.2) Матрица А транспонированная, умноженная на 2.5 и в первой строке вычтенная на 5: ")
pretty_print_matrix(matrixAt)

matrixB = numpy.random.randint(0, 11, (6, 3))
#print("\n4) Матрица В: ")
#pretty_print_matrix(matrixB)

vectorA = numpy.sum(matrixAt, axis=1)
vectorB = numpy.sum(matrixB, axis=0)
print("\n5.1) Вектор матрицы А и его размер: ")
pretty_print_matrix(vectorA)
print(vectorA.size)

print("\n5.2) Вектор матрицы В и его размер: ")
pretty_print_matrix(vectorB)
print(vectorB.size)

print("\n6) Произвидение матриц А и В: ")
pretty_print_matrix(matrixAt.dot(matrixB))

print("\n7.1) Обновленная матрица А:")
matrixA = numpy.delete(matrixAt, 3, axis=1)
pretty_print_matrix(matrixA)

for i in range(3):
    row = numpy.random.randint(10, 21, (matrixB.shape[0], 1))
    matrixB = numpy.column_stack((matrixB, row))
print("\n7.2) Обновленная матрица В: ")
pretty_print_matrix(matrixB)

print("\n8.1) Определитель матрицы A: ")
print(numpy.linalg.det(matrixA))

print("\n8.2) Определитель матрицы B: ")
print(round(numpy.linalg.det(matrixB)))

try:
    inverse_matrix = numpy.linalg.inv(matrixA)
    print("\n8.3) Обратная матрица А:")
    pretty_print_matrixP(inverse_matrix)
except:
    print("\n8.3) Обратная матрица для А не может быть вычислена, так как определитель равен нулю.")

try:
    inverse_matrix = numpy.linalg.inv(matrixB)
    print("\n8.4) Обратная матрица В:")
    pretty_print_matrixP(inverse_matrix)
except:
    print("\n8.4) Обратная матрица для В не может быть вычислена, так как определитель равен нулю.")

matrixA = numpy.linalg.matrix_power(matrixA, 6)
print("\n9.1) Матрица А в 6 степени:")
pretty_print_matrixP(matrixA)

matrixB = numpy.linalg.matrix_power(matrixB, 14)
print("\n9.2) Матрица В в 14 степени:")
pretty_print_matrixP(matrixB)

A = numpy.array([[2, -5, 1, 0],
                [1, -1, -13, 0],
                [3, -2, -2, -4],
                [4, 0, 2.7, -1.3]])
b = numpy.array([-4, 2.6, 1, -2])
print("\n10) Решение системы уравнений ")
pretty_print_matrixP(numpy.linalg.solve(A, b))



