#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数字生成模块
负责生成自然数、真分数（包括纯分数和带分数）
以及相关的数字转换和计算功能
"""
import random
from fractions import Fraction


def generate_natural_number(range_limit):
    """
    生成一个指定范围内的自然数

    参数:
        range_limit: 数值上限（不包含）

    返回:
        自然数的元组表示 (整数部分, 分子, 分母)
    """
    num = random.randint(0, range_limit - 1)
    return num, 0, 1  # 表示自然数n = n + 0/1


def generate_proper_fraction(range_limit):
    """
    生成一个指定范围内的真分数（纯分数或带分数）

    参数:
        range_limit: 数值上限（不包含）

    返回:
        分数的元组表示 (整数部分, 分子, 分母)
    """
    # 50%概率生成纯分数，50%概率生成带分数
    if random.random() < 0.5:
        # 纯分数：分子 < 分母
        denominator = random.randint(2, range_limit)
        numerator = random.randint(1, denominator - 1)
        return 0, numerator, denominator
    else:
        # 带分数：整数部分 > 0，分子 < 分母
        integer_part = random.randint(1, range_limit - 1)
        denominator = random.randint(2, range_limit)
        numerator = random.randint(1, denominator - 1)
        return integer_part, numerator, denominator


def generate_number(range_limit):
    """
    随机生成一个数字（自然数或真分数）

    参数:
        range_limit: 数值上限（不包含）

    返回:
        数字的元组表示 (整数部分, 分子, 分母)
    """
    if random.random() < 0.5:
        return generate_natural_number(range_limit)
    else:
        return generate_proper_fraction(range_limit)


def number_to_string(number):
    """
    将数字元组转换为字符串表示

    参数:
        number: 数字元组 (整数部分, 分子, 分母)

    返回:
        数字的字符串表示
    """
    integer_part, numerator, denominator = number
    if denominator == 1:
        # 自然数
        return str(integer_part)
    elif integer_part == 0:
        # 纯分数
        return f"{numerator}/{denominator}"
    else:
        # 带分数
        return f"{integer_part}'{numerator}/{denominator}"


def number_to_fraction(number):
    """
    将数字元组转换为Fraction对象以便计算

    参数:
        number: 数字元组 (整数部分, 分子, 分母)

    返回:
        对应的Fraction对象
    """
    integer_part, numerator, denominator = number
    return Fraction(integer_part * denominator + numerator, denominator)


def fraction_to_number(fraction):
    """
    将Fraction对象转换为数字元组表示

    参数:
        fraction: Fraction对象

    返回:
        数字的元组表示 (整数部分, 分子, 分母)
    """
    numerator = fraction.numerator
    denominator = fraction.denominator

    if denominator == 1:
        return numerator, 0, 1

    integer_part = numerator // denominator
    remainder = numerator % denominator

    if remainder == 0:
        return integer_part, 0, 1
    else:
        # 约分
        gcd_val = gcd(remainder, denominator)
        return integer_part, remainder // gcd_val, denominator // gcd_val







