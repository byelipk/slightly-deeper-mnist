from collections import Counter

def take_n(data, n, test_condition=lambda x: True):
    """
    Return the index position of the first N items that appear as keys
    in a "keeplist" and which also match a condition.
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
