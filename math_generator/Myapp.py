#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序入口
负责解析命令行参数并协调各模块执行相应功能
支持同时生成题目和评分
"""
import argparse
from generator import generate_problems
from grader import MathProblemGrader


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成小学四则运算题目并评分')
    parser.add_argument('-n', type=int, help='生成题目的个数')
    parser.add_argument('-r', type=int, help='数值范围（自然数、真分数和真分数分母的最大值）')
    parser.add_argument('-e', help='题目文件路径，用于评分')
    parser.add_argument('-a', help='答案文件路径，用于评分')

    args = parser.parse_args()

    # 执行生成题目功能（如果参数齐全）
    if args.n is not None and args.r is not None:
        if args.n <= 0 or args.r <= 0:
            print("错误: 题目数量和范围必须为正整数")
            return

        print(f"正在生成 {args.n} 道题目，数值范围为 {args.r}...")
        try:
            problems, answers = generate_problems(args.n, args.r)

            # 写入题目文件
            with open('Exercises.txt', 'w', encoding='utf-8') as f:
                for i, problem in enumerate(problems):
                    f.write(f"{i + 1}. {problem}\n")

            # 写入答案文件
            with open('Answers.txt', 'w', encoding='utf-8') as f:
                for i, answer in enumerate(answers):
                    f.write(f"{i + 1}. {answer}\n")

            print(f"题目已写入 Exercises.txt")
            print(f"答案已写入 Answers.txt")
        except Exception as e:
            print(f"生成题目时出错: {e}")

    # 执行评分功能（如果参数齐全）
    if args.e and args.a:
        grader = MathProblemGrader(args.e, args.a)
        grader.grade()

    # 如果没有提供任何有效参数组合，显示帮助信息
    if (args.n is None or args.r is None) and (not args.e or not args.a):
        print("错误: 参数组合无效")
        parser.print_help()


if __name__ == "__main__":
    main()
