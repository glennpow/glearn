def intersects(l1, l2):
    if not isinstance(l1, list):
        l1 = [l1]
    if not isinstance(l2, list):
        l2 = [l2]
    for v1 in l1:
        for v2 in l2:
            if v1 == v2:
                return True
    return False


def intersection(l1, l2):
    return [v for v in l1 if v in l2]


def subtraction(l1, l2):
    return [v for v in l1 if v not in l2]
