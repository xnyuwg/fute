import numpy


class UtilMath:
    def __init__(self):
        pass

    @staticmethod
    def calculate_f1(p, r):
        if p + r == 0:
            return 0
        f1 = (2 * p * r) / (p + r)
        return f1

    @staticmethod
    def calculate_precision_recall_f1(true_positive, ans_positive, pred_positive):
        precision = true_positive / pred_positive if pred_positive != 0 else 0
        recall = true_positive / ans_positive if ans_positive != 0 else 0
        f1 = UtilMath.calculate_f1(precision, recall)
        return precision, recall, f1

    @staticmethod
    def two_line_overlap(x1, x2, y1, y2) -> bool:
        assert x2 > x1 and y2 > y1
        overlap_a1 = max(x1, y1)
        overlap_a2 = min(x2, y2)
        if overlap_a1 > overlap_a2:
            return False
        overlap_length = overlap_a2 - overlap_a1
        return overlap_length

    @staticmethod
    def if_two_line_overlap_more_than_rate_to_themselves(x1, x2, y1, y2, overlap_ratio=0.5) -> bool:
        overlap_length = UtilMath.two_line_overlap(x1=x1, x2=x2, y1=y1, y2=y2)
        x_length = x2 - x1
        y_length = y2 - y1
        if overlap_length / x_length > overlap_ratio and overlap_length / y_length > overlap_ratio:
            return True
        else:
            return False

    @staticmethod
    def compute_overlap_area_of_two_rectangle_with_list(r1: list[float], r2: list[float]) -> float:
        return UtilMath.compute_overlap_area_of_two_rectangle_with_int(ax1=r1[0], ay1=r1[1], ax2=r1[2], ay2=r1[3], bx1=r2[0], by1=r2[1], bx2=r2[2], by2=r2[3])

    @staticmethod
    def compute_overlap_area_of_two_rectangle_with_int(ax1: float, ay1: float, ax2: float, ay2: float, bx1: float, by1: float, bx2: float, by2: float) -> float:
        """ (x1,y1) left top (x2, y2) right bottom """
        overlap_width = min(ax2, bx2) - max(ax1, bx1)
        overlap_height = min(ay2, by2) - max(ay1, by1)
        overlap_area = max(overlap_width, 0) * max(overlap_height, 0)
        return overlap_area

    @staticmethod
    def compute_are_of_rectangle(r: list[float]):
        area = (r[2] - r[0]) * (r[3] - r[1])
        return area

    @staticmethod
    def split_a_list_by_per_sub_len(list_len, sub_len):
        # right not include
        left = 0
        right = sub_len
        out_lens = []
        while right <= list_len:
            out_lens.append((left, right))
            left = right
            right = left + sub_len
        if left < list_len:
            out_lens.append((left, list_len))
        return out_lens

    @staticmethod
    def numpy_softmax(x):
        e_x = numpy.exp(x - numpy.max(x))
        return e_x / e_x.sum(axis=0)





