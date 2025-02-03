import math
import numpy
import scipy
import sympy
from sympy.stats.sampling.sample_numpy import numpy


# вариант 2

## задание 1
def f(x):
    return math.log(math.sqrt(x))
x0 = 2.0

firDir = scipy.misc.derivative(f, x0, dx=1e-6)
secDir = scipy.misc.derivative(f, x0, n=2, dx=1e-6)
print(f"первая производная: {firDir}")
print(f"вторая производная: {secDir}")

## задание 2
Symbol = sympy.symbols('x')
Func = sympy.ln(sympy.sqrt(Symbol))
FuncDir = sympy.diff(Func, Symbol)
print(f"функция: {Func}")
print(f"первая производная: {FuncDir}")

## задание 3
a = 1
b = 6
n = 1000

Range = numpy.linspace(a, b, n) # множество разбиение
FuncOnARange = numpy.log(numpy.sqrt(Range)) # множество значений функций
Integral = scipy.integrate.trapezoid(FuncOnARange, Range)

#integral = scipy.integrate.quad(f, a, b)

print(f"интеграл равен: {Integral}")

## задание 4

integralN = sympy.integrate(Func, Symbol)
print(f"неопределенный интеграл равен: {integralN} + C")

## задание 5

def сfunc(vars):
    x, y = vars
    return (x - 3)**2 + (y - 1)**2

def constraint(vars):
    x, y = vars
    return -2*x + y - 2

def constraint2(vars):
    x, y = vars
    return 3*y - 10

iguess = [0, 0]

constraints = [
    {'type': 'ineq', 'fun': constraint},
    {'type': 'ineq', 'fun': constraint2},
]

result = scipy.optimize.minimize(сfunc, iguess, constraints=constraints)

if result.success:
    print("птимальное значение переменных:", result.x)
    print("инимальное значение целевой функции:", result.fun)
else:
    print("птимизация не удалась")