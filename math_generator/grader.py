#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
评分模块
负责读取题目和答案文件，进行批改并生成评分结果
"""
import re
from fractions import Fraction
import number_generator as ng


class MathProblemGrader:
    """数学题目评分器类"""

    def __init__(self, exercise_file, answer_file):
        """
        初始化评分器

        参数:
            exercise_file: 题目文件路径
            answer_file: 答案文件路径
        """
        self.exercise_file = exercise_file
        self.answer_file = answer_file

    def read_problems_and_answers(self):
        """
        读取题目和答案文件

        返回:
            题目列表和答案列表
        """
        try:
            with open(self.exercise_file, 'r', encoding='utf-8') as f:
                exercises = [line.strip().split('. ', 1)[1] for line in f if line.strip()]

            with open(self.answer_file, 'r', encoding='utf-8') as f:
                answers = [line.strip().split('. ', 1)[1] for line in f if line.strip()]

            return exercises, answers
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return [], []

    @staticmethod
    def parse_expression(expr):
        """
        解析表达式并计算结果

        参数:
            expr: 表达式字符串

        返回:
            计算结果（Fraction对象）
        """
        # 替换×为*，÷为/以便计算
        expr_for_eval = expr.replace('×', '*').replace('÷', '/')

        # 处理分数
        def replace_fraction(match):
            s = match.group(0)
            if "'" in s:
                # 带分数
                integer, frac = s.split("'")
                numerator, denominator = frac.split("/")
                return f"({integer}) + ({numerator})/({denominator})"
            elif "/" in s:
                # 纯分数
                numerator, denominator = s.split("/")
                return f"({numerator})/({denominator})"
            else:
                # 自然数
                return s

        # 匹配分数格式：带分数（如3'1/2）、纯分数（如1/2）或自然数（如5）
        pattern = r'\d+\'\d+/\d+|\d+/\d+|\d+'
        expr_for_eval = re.sub(pattern, replace_fraction, expr_for_eval)

        # 使用Fraction计算，确保精确性
        try:
            # 使用安全的eval环境
            result = eval(expr_for_eval, {"__builtins__": None}, {"Fraction": Fraction})
            if not isinstance(result, Fraction):
                result = Fraction(result)
            return result
        except Exception as e:
            print(f"解析表达式时出错: {expr}，错误: {e}")
            return None

    def grade(self):
        """
        批改答案并生成统计结果
        结果将写入Grade.txt文件
        """
        exercises, answers = self.read_problems_and_answers()

        if len(exercises) != len(answers):
            print("错误: 题目数量和答案数量不匹配")
            return

        correct = []
        wrong = []

        for i in range(len(exercises)):
            # 提取表达式（去掉等号）
            expr = exercises[i].rsplit(' =', 1)[0]

            # 解析表达式并计算正确答案
            result = MathProblemGrader.parse_expression(expr)
            if result is None:
                wrong.append(i + 1)
                continue

            correct_answer = ng.fraction_to_number(result)
            correct_answer_str = ng.number_to_string(correct_answer)

            # 比较答案
            if correct_answer_str == answers[i]:
                correct.append(i + 1)  # 题目编号从1开始
            else:
                wrong.append(i + 1)

        # 写入评分结果
        with open('Grade.txt', 'w', encoding='utf-8') as f:
            f.write(f"Correct: {len(correct)} ({', '.join(map(str, correct))})\n")
            f.write(f"Wrong: {len(wrong)} ({', '.join(map(str, wrong))})\n")



