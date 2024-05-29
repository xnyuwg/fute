import re
import hashlib


class UtilString:
    def __init__(self):
        pass

    @staticmethod
    def pascal_case_to_snake_case(camel_case: str):
        """大驼峰（帕斯卡）转蛇形"""
        snake_case = re.sub(r"(?P<key>[A-Z])", r"_\g<key>", camel_case)
        return snake_case.lower().strip('_')

    @staticmethod
    def snake_case_to_pascal_case(snake_case: str):
        """蛇形转大驼峰（帕斯卡）"""
        words = snake_case.split('_')
        return ''.join(word.title() for word in words)

    @staticmethod
    def character_tokenize(s: str):
        tokens = []
        for c in s:
            tokens.append(c)
        return tokens

    @staticmethod
    def if_str_contain_alpha(s: str) -> bool:
        for c in s:
            if c.isalpha():
                return True
        return False

    @staticmethod
    def longest_common_subsequence_length_number(text1, text2) -> int:
        # If text1 doesn't reference the shortest string, swap them.
        if len(text2) < len(text1):
            text1, text2 = text2, text1

        # The previous and current column starts with all 0's and like
        # before is 1 more than the length of the first word.
        previous = [0] * (len(text1) + 1)
        current = [0] * (len(text1) + 1)

        # Iterate up each column, starting from the last one.
        for col in reversed(range(len(text2))):
            for row in reversed(range(len(text1))):
                if text2[col] == text1[row]:
                    current[row] = 1 + previous[row + 1]
                else:
                    current[row] = max(previous[row], current[row + 1])
            # The current column becomes the previous one, and vice versa.
            previous, current = current, previous

        # The original problem's answer is in previous[0]. Return it.
        return previous[0]

    @staticmethod
    def longest_common_substring_length_number(X, Y) -> int:
        # Create a table to store lengths of
        # longest common suffixes of substrings.
        # Note that LCSuff[i][j] contains the
        # length of longest common suffix of
        # X[0...i-1] and Y[0...j-1]. The first
        # row and first column entries have no
        # logical meaning, they are used only
        # for simplicity of the program.

        # LCSuff is the table with zero
        # value initially in each cell
        m = len(X)
        n = len(Y)

        LCSuff = [[0 for k in range(n + 1)] for l in range(m + 1)]

        # To store the length of
        # longest common substring
        result = 0

        # Following steps to build
        # LCSuff[m+1][n+1] in bottom up fashion
        for i in range(m + 1):
            for j in range(n + 1):
                if (i == 0 or j == 0):
                    LCSuff[i][j] = 0
                elif (X[i - 1] == Y[j - 1]):
                    LCSuff[i][j] = LCSuff[i - 1][j - 1] + 1
                    result = max(result, LCSuff[i][j])
                else:
                    LCSuff[i][j] = 0
        return result

    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split('(\d+)', s)]

    @staticmethod
    def capital_ratio(s):
        total_letters = sum(c.isalpha() for c in s)
        uppercase_letters = sum(1 for c in s if c.isupper())
        # If there are no letters in the string, return 0
        if total_letters == 0:
            return 0
        return uppercase_letters / total_letters

    @staticmethod
    def str_to_bool(s):
        s = s.lower()
        if s in ['true', 't', 'yes', 'y', '1']:
            return True
        elif s in ['false', 'f', 'no', 'n', '0']:
            return False
        else:
            return None

    @staticmethod
    def number_string_to_number_list(s):
        # Split the string by commas
        parts = s.split(',')
        numbers = []
        for part in parts:
            # Check if part contains a dash, indicating a range
            if '-' in part:
                # Split the range into start and end
                start, end = map(int, part.split('-'))
                # Add all numbers in this range to the list
                numbers.extend(range(start, end + 1))
            else:
                # Add the single number to the list
                numbers.append(int(part))
        return numbers

    @staticmethod
    def compute_md5(data: str) -> str:
        md5 = hashlib.md5()
        md5.update(data.encode('utf-8'))
        return md5.hexdigest()
