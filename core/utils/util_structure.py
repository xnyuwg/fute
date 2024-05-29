import logging
from collections import Counter
import itertools
import random


class UtilStructure:
    def __init__(self):
        pass

    @staticmethod
    def find_most_common_element_in_list(x):
        if len(x) == 0:
            return None
        counts = Counter(x)
        max_count = counts.most_common(1)[0][1]
        out = [value for value, count in counts.most_common() if count == max_count]
        return out

    @staticmethod
    def delete_elements_in_list(input_list: list, elements_to_remove: list) -> list:
        for x in elements_to_remove:
            while x in input_list:
                input_list.remove(x)
        return input_list

    @staticmethod
    def find_max_number_index(x: list):
        res = x.index(max(x))
        return res

    @staticmethod
    def find_max_and_number_index(x: list):
        max_number = max(x)
        index = x.index(max_number)
        return max_number, index

    @staticmethod
    def new_element_count_for_dict(dic: dict, element, added_count=1):
        if element in dic:
            dic[element] += added_count
        else:
            dic[element] = added_count

    @staticmethod
    def dict_count_update_a2b(a: dict, b: dict):
        for k, v in a.items():
            if k in b:
                b[k] += v
            else:
                b[k] = v

    @staticmethod
    def average_value_of_digital_key_and_count_value(dic: dict):
        total = 0
        count = 0
        for k, v in dic.items():
            total += k * v
            count += v
        if count == 0:
            logging.warning('average_value_of_digital_key_and_count_value count is 0 !, with dic: {}'.format(dic))
            return 0
        else:
            return total / count

    @staticmethod
    def list_append(x: list, element) -> list:
        if x is None:
            x = [element]
        else:
            x.index(element)
        return x

    @staticmethod
    def find_max_element_according_to_key(x: list, key_fn):
        max_value = None
        max_element = None
        for e in x:
            if max_value is None or key_fn(e) > max_value:
                max_value = key_fn(e)
                max_element = e
        return max_element

    @staticmethod
    def find_max_elements_according_to_key(x: list, key_fn):
        max_value = None
        max_elements = None
        for e in x:
            if max_value is None or key_fn(e) > max_value:
                max_value = key_fn(e)
                max_elements = [e]
            elif key_fn(e) == max_value:
                max_elements.append(e)
        return max_elements

    @staticmethod
    def get_ratio_dict_from_count_dict(dic):
        ratio = {}
        max_length = max(list(dic.keys())) + 1
        now_count = 0
        for i in range(max_length + 1):
            if i in dic:
                now_count += dic[i]
            ratio[i] = now_count
        for k in ratio:
            ratio[k] = ratio[k] / now_count
        return ratio

    @staticmethod
    def count_dict_update(dic, key, value=1):
        if key in dic:
            dic[key] += value
        else:
            dic[key] = value
        return dic

    @staticmethod
    def find_sorted_key_value_tuple_in_dict(dic: dict, default=[(0, 0)]):
        max_value = None
        max_elements = None
        for k, v in dic.items():
            if max_value is None:
                max_value = v
                max_elements = [(k, v)]
            elif v > max_value:
                max_value = v
                max_elements = [(k, v)]
            elif v == max_value:
                max_elements.append((k, v))
        if max_elements is None:
            max_elements = default
        if len(max_elements) > 1:
            max_elements = sorted(max_elements, key=lambda x: x[0], reverse=True)
        return max_elements

    @staticmethod
    def if_one_of_element_of_alist_in_blist(alist, blist):
        for a in alist:
            if a in blist:
                return True
        return False

    @staticmethod
    def if_any_element_of_alist_in_blist(alist, blist):
        for a in alist:
            if a not in blist:
                return False
        return True

    @staticmethod
    def get_count_dict_sorted(count_dic, reverse=False):
        count_dic = [(k, v) for k, v in count_dic.items()]
        count_dic = sorted(count_dic, key=lambda x: x[1], reverse=reverse)
        return count_dic

    @staticmethod
    def if_two_list_has_common_element(listA, listB):
        return bool(set(listA) & set(listB))

    @staticmethod
    def generate_iter_lists(x_list, repeat_times):
        return list(itertools.product(x_list, repeat=repeat_times))

    @staticmethod
    def random_select_from_list(x_list, sel_num, change_order=True):
        if change_order:
            shuffled_list = x_list[:]
            random.shuffle(shuffled_list)
            return shuffled_list[:sel_num]
        else:
            indices = sorted(random.sample(range(len(x_list)), min(sel_num, len(x_list))))
            return [x_list[i] for i in indices]
