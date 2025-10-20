#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
表达式处理模块
负责表达式的分词、标准化和去重处理
"""
import re


def tokenize_expression(expr):
    """
    将表达式拆分为 tokens

    参数:
        expr: 表达式字符串

    返回:
        拆分后的tokens列表
    """
    tokens = re.findall(r'\d+\'\d+/\d+|\d+/\d+|\d+|[-+×÷()]', expr)
    return tokens


def normalize_expression(expr):
    """
    标准化表达式，用于判断是否重复

    参数:
        expr: 表达式字符串

    返回:
        标准化后的表达式字符串
    """
    tokens = tokenize_expression(expr)
    normalized = normalize_tokens(tokens)
    return ' '.join(normalized)


def normalize_tokens(tokens):
    """
    标准化 tokens 列表，处理交换律和括号

    参数:
        tokens: 表达式的tokens列表

    返回:
        标准化后的tokens列表
    """
    # 处理括号
    if len(tokens) == 3 and tokens[0] == '(' and tokens[-1] == ')':
        return normalize_tokens(tokens[1:-1])

    # 查找最外层的运算符（考虑括号）
    op_pos = -1
    paren_count = 0

    for i, token in enumerate(tokens):
        if token == '(':
            paren_count += 1
        elif token == ')':
            paren_count -= 1
        elif paren_count == 0 and token in ['+', '×']:
            op_pos = i

    if op_pos != -1:
        # 对于加法和乘法，递归标准化左右两边并确保左边 <= 右边
        op = tokens[op_pos]
        left = normalize_tokens(tokens[:op_pos])
        right = normalize_tokens(tokens[op_pos + 1:])

        if compare_token_lists(left, right) > 0:
            left, right = right, left

        return left + [op] + right
    else:
        # 对于减法和除法，只递归标准化，不交换
        for i, token in enumerate(tokens):
            if token == '(':
                paren_count += 1
            elif token == ')':
                paren_count -= 1
            elif paren_count == 0 and token in ['-', '÷']:
                op = token
                left = normalize_tokens(tokens[:i])
                right = normalize_tokens(tokens[i + 1:])
                return left + [op] + right

    # 如果没有运算符，直接返回
    return tokens


def compare_token_lists(list1, list2):
    """
    比较两个 token 列表，用于标准化

    参数:
        list1, list2: 要比较的两个tokens列表

    # 先比较长度
    返回:
        -1: list1 < list2
        0: list1 == list2
        1: list1 > list2
    """
    if len(list1) < len(list2):
        return -1
    elif len(list1) > len(list2):
        return 1

    # 再逐个比较元素
    for t1, t2 in zip(list1, list2):
        if t1 < t2:
            return -1
        elif t1 > t2:
 

