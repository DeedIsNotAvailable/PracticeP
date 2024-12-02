import math
import numpy
import scipy
import sympy as sp
from sympy.stats.sampling.sample_scipy import scipy


# вариант 2

def f(x):
    return math.log(math.sqrt(x))
x0 = 2.0
firDir = scipy.misc.derivative(f, x0, dx=1e-6)
secDir = scipy.misc.derivative(f, x0, n=2, dx=1e-6)
print(f"первая производная {firDir}")
print(f"вторая производная {secDir}")

Symbol = sp.symbols('x')
Func = sp.ln(sp.sqrt(Symbol))
FuncDir = Func.diff(Symbol)
print(f"функция: {Func}")
print(f"первая производная: {FuncDir}")

diapoz = numpy.linspace(1, 6, 1000)
Func2 = numpy.log(numpy.sqrt(diapoz))
integral = scipy.integrate.trapezoid(Func2, diapoz)
print(f"интеграл равен: {integral}")

integralNeopr = sp.integrate(Func, Symbol)
print(f"неопределенный интеграл равен: {integralNeopr} + C")

def objective_func(vars):
    x, y = vars
    return (x - 3)**2 + y

def constraint(vars):
    x, y = vars
    return -2*x + y - 2

def constraint2(vars):
    x, y = vars
    return 3*y - 10

initial_guess = [0, 0]

constraints = [
    {'type': 'ineq', 'fun': constraint},
    {'type': 'ineq', 'fun': constraint2},
]

result = scipy.optimize.minimize(objective_func, initial_guess, constraints=constraints)

if result.success:
    print("Оптимальное значение переменных:", result.x)
    print("Минимальное значение целевой функции:", result.fun)
else:
    print("Оптимизация не удалась")