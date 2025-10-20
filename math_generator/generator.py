#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
表达式生成模块
负责生成符合要求的四则运算表达式
"""
import random
import number_generator as ng
import expression_processor as ep


def generate_expression(range_limit, max_operators=3, generated_expressions=None):
    """
    生成一个表达式，返回表达式字符串和结果

    参数:
        range_limit: 数值范围上限
        max_operators: 最大运算符数量
        generated_expressions: 已生成的表达式集合，用于去重

    返回:
        表达式字符串和结果的元组
    """
    # 初始化已生成表达式集合（如果未提供）
    if generated_expressions is None:
        generated_expressions = set()

    # 30%概率生成一个数字（终止递归）
    if max_operators == 0 or random.random() < 0.3:
        number = ng.generate_number(range_limit)
        return ng.number_to_string(number), number

    # 生成两个子表达式
    op_count = random.randint(1, max_operators)
    left_ops = random.randint(0, op_count - 1)
    right_ops = op_count - 1 - left_ops

    left_expr, left_val = generate_expression(range_limit, left_ops, generated_expressions)
    right_expr, right_val = generate_expression(range_limit, right_ops, generated_expressions)

    # 随机选择运算符
    operator = random.choice(['+', '-', '×', '÷'])

    # 确保减法和除法的有效性
    left_frac = ng.number_to_fraction(left_val)
    right_frac = ng.number_to_fraction(right_val)

    valid = False
    result_frac = None

    if operator == '-':
        if left_frac >= right_frac:
            result_frac = left_frac - right_frac
            valid = True
    elif operator == '÷':
        if right_frac != 0:
            result_frac = left_frac / right_frac
            # 检查结果是否为真分数
            if result_frac <= 1:
                valid = True
    else:  # '+' 或 '×'
        if operator == '+':
            result_frac = left_frac + right_frac
        else:  # '×'
            result_frac = left_frac * right_frac
        valid = True

    if not valid:
        # 生成的表达式无效，重新生成
        return generate_expression(range_limit, max_operators, generated_expressions)

    result_val = ng.fraction_to_number(result_frac)

    # 随机添加括号（只有当子表达式有运算符时才可能添加）
    add_left_paren = random.random() < 0.3 and left_ops > 0
    add_right_paren = random.random() < 0.3 and right_ops > 0

    if add_left_paren:
        left_expr = f"({left_expr})"
    if add_right_paren:
        right_expr = f"({right_expr})"

    # 构建表达式字符串
    expr = f"{left_expr} {operator} {right_expr}"

    # 检查是否重复
    normalized_expr = ep.normalize_expression(expr)
    if normalized_expr in generated_expressions:
        return generate_expression(range_limit, max_operators, generated_expressions)

    generated_expressions.add(normalized_expr)
    return expr, result_val


def generate_problems(num_problems, range_limit):
    """
    生成指定数量的题目

    参数:
        num_problems: 题目数量
        range_limit: 数值范围上限

    返回:
        题目列表和答案列表
    """
    problems = []
    answers = []
    generated_expressions = set()

    for _ in range(num_problems):
        # 最多尝试100次生成不重复的题目
        for _ in range(100):
            expr, result = generate_expression(range_limit, 3, generated_expressions)
            if expr:
                break
        else:
            raise Exception("无法生成足够的不重复题目，请尝试减小题目数量或增大范围")



