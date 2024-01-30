# 生成基础的ASCII字符集（英文和数字）
base_chars = [chr(i) for i in range(32, 127)]

# 生成中文字符集（以Unicode范围为例）
chinese_chars = [chr(i) for i in range(0x4E00, 0x9FFF+1)]

# 合并字符集
char_set = base_chars + chinese_chars
char_set.append('·')
# print(char_set)
# # 打印字符集大小
# print("字符集大小:", len(char_set))
# print(char_set[10])