"""
В этом файле собраны функции из блокнота.
Они реализуют fft "как надо" - в радианных координатах, основываясь на библиотечных рутинах.
"""
import numpy as np
from numpy import fft, pi, linspace


def make_rad_fft(f_x, d):
    """
    Вычисляет преобразование Фурье без двух "пи" в показателе экспоненты

    Параметры:

    f_x - массив значений функции, задаваемый "как есть", без особенностей хранения
    и определённый на диапазоне [-M*dx/2, M*dx/2] в M = 2**N равноудалённых точках, без последней.

    d = dx - желаемый шаг по переменной x

    Области определения:

    xi_space = np.linspace( -pi/dx, pi/dx, num = M, endpoint = False)
    x_space = np.linspace( -M*dx/2, M*dx/2, num = M, endpoint = False)
    """
    dx = d
    M = len(f_x)

    sign_change_k = np.array([(-1)**k for k in range(0, M)])
    sign_change_l = np.array([(-1)**l for l in range(0, M)])
    # учитываем порядок хранения
    sign_change_l = fft.fftshift(sign_change_l)

    f = sign_change_k * f_x
    f_hat = dx * sign_change_l * fft.fft(f)

    # избегаем особенностей хранения результатов fft, нам они не нужны.
    return f_hat


def make_rad_ifft(f_hat_xi, d):
    """
    Вычисляет обратное преобразование Фурье без двух "пи" в показателе экспоненты

    Параметры:

    f_xi - массив значений функции, задаваемый "как есть", без особенностей хранения
    и определённый на диапазоне [-pi/d, pi/d] в M = 2**N равноудалённых точках, без последней.

    d = dx - желаемый шаг по переменной x, после того, как ifft отработает

    Области определения:

    xi_space = np.linspace( -pi/dx, pi/dx, num = M, endpoint = False)
    x_space = np.linspace( -M*dx/2, M*dx/2, num = M, endpoint = False)
    """
    dx = d
    M = len(f_hat_xi)

    sign_change_k = np.array([(-1)**k for k in range(0, M)])
    sign_change_l = np.array([(-1)**l for l in range(0, M)])

    f = (1/dx) * sign_change_k * fft.ifft(sign_change_l * f_hat_xi)
    return f


def make_fft_spaces(M, dx):

    x_space = linspace(-M * dx / 2, M * dx / 2, num=M, endpoint=False)
    u_space = linspace(-pi / dx, pi / dx, num=M, endpoint=False)
    du = u_space[1] - u_space[0]

    # print("В вычислениях будет использовано всего", M, "точек")
    # print("du = ", du, "Частота меняется от ", -pi / dx, "до", pi / dx)
    #
    # print("dx = ", dx, "Пространственная переменная меняется от ", x_space[0], "до", x_space[M - 1])
    #
    # print("Отношение длин диапазонов пространство/частота:", (x_space[M - 1] - x_space[0]) / (2 * pi / dx))
    # print("Отношение длин диапазонов частота/пространство:", (2 * pi / dx) / (x_space[M - 1] - x_space[0]))

    return x_space, u_space
