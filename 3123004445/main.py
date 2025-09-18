import sys
import os
from plagiarism_utils import (
    validate_file_path,
    read_file,
    preprocess_text,
    calculate_similarity
)


def main():
    """
    主函数：执行论文查重的完整流程

    该函数负责：
    1. 解析命令行参数
    2. 验证文件路径的有效性
    3. 读取并预处理原文和抄袭版文本
    4. 计算综合重复率
    5. 将结果输出到文件和控制台
    """
    try:
        # 检查命令行参数数量是否为4（脚本名 + 3个参数：原文路径、抄袭版路径、结果路径）
        if len(sys.argv) != 4:
            raise ValueError("参数错误！\n正确格式：python main.py [原文路径] [抄袭版路径] [结果路径]")

        # 解析命令行参数，获取三个文件的路径
        original_path = sys.argv[1]  # 原文文件路径
        plagiarized_path = sys.argv[2]  # 抄袭版文件路径
        result_path = sys.argv[3]  # 结果输出文件路径

        # 验证原文文件路径的有效性（存在性、是否为文件、读取权限、大小限制等）
        original_path = validate_file_path(original_path, "原文")
        # 验证抄袭版文件路径的有效性
        plagiarized_path = validate_file_path(plagiarized_path, "抄袭版论文")

        # 验证结果文件所在目录是否存在，若不存在则创建（递归创建多级目录）
        output_dir = os.path.dirname(result_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)  # exist_ok=True 避免目录已存在时报错
            except Exception as e:
                raise IOError(f"无法创建输出目录: {output_dir}, 错误原因: {str(e)}")

        # 验证结果文件是否有写入权限（若文件已存在）
        if os.path.exists(result_path) and not os.access(result_path, os.W_OK):
            raise PermissionError(f"没有结果文件的写入权限: {result_path}")

        # 读取原文内容（自动尝试多种编码以避免乱码）
        original_text = read_file(original_path)
        # 读取抄袭版论文内容
        plagiarized_text = read_file(plagiarized_path)

        # 预处理文本：生成两种格式的结果
        # original_tokens：分词列表（用于词频匹配、余弦相似度）
        # original_str：清洗后的字符串（用于编辑距离计算）
        original_tokens, original_str = preprocess_text(original_text)
        plagiarized_tokens, plagiarized_str = preprocess_text(plagiarized_text)

        # 检查预处理结果是否有效（避免空内容导致后续计算出错）
        if not original_tokens or not original_str:
            raise ValueError("原文预处理后无有效内容，可能是文件为空或仅含空白字符")
        if not plagiarized_tokens or not plagiarized_str:
            raise ValueError("抄袭版论文预处理后无有效内容，可能是文件为空或仅含空白字符")

        # 计算综合相似度（融合词频匹配、余弦相似度、编辑距离三种算法的结果）
        similarity = calculate_similarity(original_tokens, plagiarized_tokens, original_str, plagiarized_str)

        # 将相似度转换为百分比，并保留两位小数
        result = round(similarity * 100, 2)

        # 将结果写入文件，并在控制台输出
        try:
            with open(result_path, 'w', encoding='utf-8') as f:
                f.write(f"{result:.2f}")  # 写入两位小数的百分比结果
            print(f"查重完成，综合重复率: {result:.2f}%，结果已保存到 {result_path}")
        except Exception as e:
            raise IOError(f"写入结果文件失败: {str(e)}")

    except Exception as e:
        # 捕获所有异常，打印错误信息并以错误码1退出程序
        print(f"程序执行过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # 程序入口，当直接运行该脚本时执行main函数
    main()
