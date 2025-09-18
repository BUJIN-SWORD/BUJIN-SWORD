import os
import re
import jieba
import math
import string
from typing import List, Tuple
from collections import defaultdict
from synonym_database import get_synonyms


def is_chinese_char(char: str) -> bool:
    """
    判断输入字符是否为中文字符

    参数:
        char: 待判断的字符（必须是单个字符）

    返回:
        bool: 是中文字符返回True，否则返回False
    """
    # 中文字符的Unicode范围：\u4e00-\u9fff
    return len(char) == 1 and '\u4e00' <= char <= '\u9fff'


def validate_file_path(file_path: str, file_type: str) -> str:
    """
    验证文件路径的有效性，包括存在性、是否为文件、读写权限和大小限制

    参数:
        file_path: 待验证的文件路径
        file_type: 文件类型描述（用于错误提示，如"原文"、"抄袭文本"）

    返回:
        str: 规范化后的文件路径

    异常:
        ValueError: 路径为空或文件为空
        FileNotFoundError: 文件不存在
        IsADirectoryError: 路径指向目录而非文件
        PermissionError: 没有读取权限
        IOError: 文件大小超过100MB限制
    """
    # 检查路径是否为空
    if not file_path:
        raise ValueError(f"{file_type}路径不能为空")

    # 处理路径中的用户目录（~）和环境变量
    file_path = os.path.expanduser(file_path)
    file_path = os.path.expandvars(file_path)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_type}文件不存在: {file_path}")

    # 检查是否为文件（而非目录）
    if not os.path.isfile(file_path):
        raise IsADirectoryError(f"{file_type}路径不是文件: {file_path}")

    # 检查是否有读取权限
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"没有{file_type}文件的读取权限: {file_path}")

    # 获取文件大小（字节）
    file_size = os.path.getsize(file_path)

    # 检查文件是否为空
    if file_size == 0:
        raise ValueError(f"{file_type}文件为空: {file_path}")

    # 检查文件大小是否超过100MB限制
    if file_size > 100 * 1024 * 1024:  # 100MB = 100 * 1024KB * 1024B
        raise IOError(f"{file_type}文件过大（超过100MB）: {file_path}")

    return file_path


def read_file(file_path: str) -> str:
    """
    读取文件内容，自动尝试多种编码格式以解决中文乱码问题

    参数:
        file_path: 待读取的文件路径

    返回:
        str: 文件内容（已去除首尾空白）

    异常:
        Exception: 读取失败或文件内容仅含空白字符
    """
    try:
        # 常见文本编码列表，按优先级排序
        encodings = [
            'utf-8',  # 最常用的Unicode编码
            'utf-8-sig',  # 带BOM的UTF-8
            'gbk',  # 简体中文编码
            'gb2312',  # 简化版GBK
            'ISO-8859-1',  # 西欧编码
            'latin-1'  # ISO-8859-1的别名
        ]

        # 尝试多种编码读取文件
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as file_handle:
                    content = file_handle.read().strip()

                    # 检查内容是否仅包含空白字符（空格、换行等）
                    if not content.replace('\r', '').replace('\n', '').replace(' ', ''):
                        raise ValueError(f"{file_path}内容为空（仅含空白字符）")

                    return content
            except UnicodeDecodeError:
                # 编码不匹配时尝试下一种编码
                continue

        # 所有编码都尝试失败
        raise Exception(f"无法解析文件编码: {file_path}")

    except Exception as err:
        # 包装异常信息，方便上层调用追踪
        raise Exception(f"读取文件错误: {str(err)}") from err


def preprocess_text(text: str) -> Tuple[List[str], str]:
    """
    对文本进行预处理，生成两种格式的结果：
    1. 分词后的词语列表（用于词频和余弦相似度计算）
    2. 清洗后的字符串（用于编辑距离计算）

    参数:
        text: 原始文本内容

    返回:
        Tuple[List[str], str]:
            - 第一个元素：分词后的词语列表（过滤空值）
            - 第二个元素：去除标点和空格的纯文本字符串
    """
    if not text:
        return [], ""

    # --------------------------
    # 1. 处理用于编辑距离的字符串
    # --------------------------
    # 定义需要去除的标点符号（中英文标点）
    punctuation = string.punctuation + '！？。，、；：“”‘’（）【】《》'

    # 创建标点符号映射表（用于删除标点）
    translator = str.maketrans('', '', punctuation)

    # 去除标点和所有空白字符（空格、换行等）
    edit_str = text.translate(translator)
    edit_str = re.sub(r'\s+', '', edit_str)  # 去除所有空白

    # --------------------------
    # 2. 处理用于分词的词语列表
    # --------------------------
    tokens: List[str] = []  # 存储初步分割的tokens
    temp_english: List[str] = []  # 临时存储英文单词字符

    # 遍历文本中的每个字符，分离中英文
    for char in text:
        # 处理英文和数字（转为小写）
        if char.isalnum() and not is_chinese_char(char):
            temp_english.append(char.lower())
        else:
            # 遇到非英文/数字字符时，保存积累的英文单词
            if temp_english:
                english_word = ''.join(temp_english)
                tokens.append(english_word)
                temp_english = []

            # 只保留中文字符（忽略其他符号）
            if is_chinese_char(char):
                tokens.append(char)

    # 保存最后一个英文单词
    if temp_english:
        english_word = ''.join(temp_english)
        tokens.append(english_word)

    # --------------------------
    # 3. 对中文部分进行精确分词
    # --------------------------
    chinese_tokens: List[str] = []  # 最终分词结果
    temp_chinese: List[str] = []  # 临时存储中文片段

    for current_token in tokens:
        # 判断当前token是否包含中文字符
        if any(is_chinese_char(c) for c in current_token):
            temp_chinese.append(current_token)
        else:
            # 遇到英文时，先处理积累的中文片段
            if temp_chinese:
                # 使用jieba进行中文分词（精确模式）
                segmented = jieba.cut(''.join(temp_chinese), cut_all=False)
                chinese_tokens.extend(list(segmented))
                temp_chinese = []

            # 直接添加英文token
            chinese_tokens.append(current_token)

    # 处理剩余的中文片段
    if temp_chinese:
        segmented = jieba.cut(''.join(temp_chinese), cut_all=False)
        chinese_tokens.extend(list(segmented))

    # 过滤空字符串token
    final_tokens = [token for token in chinese_tokens if token.strip()]

    return final_tokens, edit_str


def word_frequency_match(original: List[str], plagiarized: List[str]) -> float:
    """
    词频精确匹配算法：统计抄袭文本中与原文相同词语的出现频率

    参数:
        original: 原文分词后的词语列表
        plagiarized: 抄袭文本分词后的词语列表

    返回:
        float: 匹配得分（0~1之间，1表示完全匹配）
    """
    # 处理空输入
    if not original or not plagiarized:
        return 0.0

    # 统计原文中每个词语的出现次数
    original_counts = defaultdict(int)
    for word in original:
        original_counts[word] += 1

    # 统计抄袭文本中与原文匹配的词语数量
    matched_count = 0
    for word in plagiarized:
        if original_counts[word] > 0:
            matched_count += 1
            original_counts[word] -= 1  # 避免重复匹配

    # 计算匹配得分（匹配数 / 抄袭文本总词数）
    return matched_count / len(plagiarized)


def cosine_similarity_score(original: List[str], plagiarized: List[str]) -> float:
    """
    带同义词扩展的余弦相似度算法：
    通过同义词扩展构建词向量，计算两个文本向量的余弦夹角

    参数:
        original: 原文分词后的词语列表
        plagiarized: 抄袭文本分词后的词语列表

    返回:
        float: 相似度得分（0~1之间，1表示完全相似）
    """
    # 处理空输入
    if not original or not plagiarized:
        return 0.0

    # --------------------------
    # 1. 构建包含同义词的联合词汇表
    # --------------------------
    vocab_set = set()
    for outer_token in original + plagiarized:
        # 将当前词及其所有同义词加入词汇表
        vocab_set.update(get_synonyms(outer_token))
    vocab = list(vocab_set)  # 转换为列表用于生成向量索引

    # --------------------------
    # 2. 生成词频向量（包含同义词扩展）
    # --------------------------
    def get_vector(tokens: List[str]) -> List[int]:
        """生成文本的词频向量（考虑同义词）"""
        word_counts = defaultdict(int)

        # 统计每个词及其同义词的出现次数
        for token in tokens:
            for synonym in get_synonyms(token):
                word_counts[synonym] += 1

        # 生成向量（按词汇表顺序）
        return [word_counts.get(word, 0) for word in vocab]

    # 生成原文和抄袭文本的向量
    original_vector = get_vector(original)
    plagiarized_vector = get_vector(plagiarized)

    # --------------------------
    # 3. 计算余弦相似度
    # --------------------------
    # 计算点积
    dot_product = sum(a * b for a, b in zip(original_vector, plagiarized_vector))

    # 计算向量模长
    norm_original = math.sqrt(sum(x ** 2 for x in original_vector))
    norm_plagiarized = math.sqrt(sum(x ** 2 for x in plagiarized_vector))

    # 避免除以零
    if norm_original == 0 or norm_plagiarized == 0:
        return 0.0

    # 余弦相似度公式：dot_product / (|a| * |b|)
    return dot_product / (norm_original * norm_plagiarized)


def edit_distance_similarity(original: str, plagiarized: str) -> float:
    """
    编辑距离相似度：计算将一个字符串转换为另一个所需的最少编辑操作（插入、删除、替换）
    相似度 = 1 - (编辑距离 / 最长字符串长度)

    参数:
        original: 原文清洗后的字符串
        plagiarized: 抄袭文本清洗后的字符串

    返回:
        float: 相似度得分（0~1之间，1表示完全相同）
    """
    # 处理空输入
    if not original or not plagiarized:
        return 0.0

    # 获取字符串长度
    len_original = len(original)
    len_plagiarized = len(plagiarized)

    # 创建动态规划(DP)表，存储子问题的编辑距离
    # dp[i][j] 表示 original[0..i-1] 到 plagiarized[0..j-1] 的编辑距离
    dp = [[0] * (len_plagiarized + 1) for _ in range(len_original + 1)]

    # 初始化边界：空字符串到长度为i/j的字符串的编辑距离
    for i in range(len_original + 1):
        dp[i][0] = i  # 需要i次删除操作
    for j in range(len_plagiarized + 1):
        dp[0][j] = j  # 需要j次插入操作

    # 填充DP表
    for i in range(1, len_original + 1):
        for j in range(1, len_plagiarized + 1):
            # 字符相同则不需要编辑
            if original[i - 1] == plagiarized[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 取三种操作的最小值 + 1（当前操作）
                dp[i][j] = 1 + min(
                    dp[i - 1][j],  # 删除
                    dp[i][j - 1],  # 插入
                    dp[i - 1][j - 1]  # 替换
                )

    # 计算最长字符串长度
    max_length = max(len_original, len_plagiarized)

    # 计算相似度（1 - 归一化编辑距离）
    return 1 - (dp[len_original][len_plagiarized] / max_length)


def calculate_similarity(
        original_tokens: List[str],
        plagiarized_tokens: List[str],
        original_str: str,
        plagiarized_str: str
) -> float:
    """
    综合三种算法的结果，计算最终的文本相似度得分

    参数:
        original_tokens: 原文分词列表
        plagiarized_tokens: 抄袭文本分词列表
        original_str: 原文清洗后的字符串
        plagiarized_str: 抄袭文本清洗后的字符串

    返回:
        float: 综合相似度得分（0~1之间）
    """





