from collections import Counter

def take_n(data, n, test_condition=lambda x: True):
    """
    Return index position of first N items that match a test condition.

    Parameters
    ==========

    :data: An enumerable, such as a list.
    :n: The number of times we can take an item.
    :test_condition: Each item we keep must meet the condition.

    Summary
    =======

    Sometimes you want to filter a list based on a condition
    and you have an upper bound of how many times you can
    take an item from a list. This function helps you
    with this problem.

    For example, if you only wanted to process integers less
    than three and could only sample each integer twice:

        data = [1,2,1,4,5,2,2]
        idx  = list(take_n(data, 2, lambda x: x < 3))
        idx  == [0,1,2,5]

        ints = data[idx]
        ints == [1,2,1,2]

    """
    keeplist = Counter()

    for idx, item in enumerate(data):

        # Use the condition to filter out unwanted items
        if test_condition(item):

            # Skip if we're already at the limit for that item.
            # Otherwise, yield the index position of the item
            # and increment the counter.
            if keeplist[item] >= n:
                continue
            else:
                keeplist[item] += 1
                yield idx
