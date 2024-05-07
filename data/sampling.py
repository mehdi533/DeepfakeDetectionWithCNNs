import random
from collections import defaultdict
import random

def downsampling(data_list):

    class_counts = defaultdict(int)

    for label, _ in data_list:
        class_counts[label] += 1
    
    min_samples = min(class_counts.values())

    balanced_data_list = []

    for label in class_counts:
        # Extract all samples of a class
        class_samples = [item for item in data_list if item[0] == label]
        # Shuffle to add randomness
        random.shuffle(class_samples)
        # Add equal number of samples from each class
        balanced_data_list.extend(class_samples[:min_samples])

    # Shuffle the final list to mix classes
    random.shuffle(balanced_data_list)

    class_counts = defaultdict(int)

    for label, _ in balanced_data_list:
        class_counts[label] += 1

    return balanced_data_list


def multiply_class(data_list, label_up, multiplier=2):

    class_counts = defaultdict(int)

    for label, _ in data_list:
        class_counts[label] += 1
    
    num_needed = class_counts[label_up]

    balanced_data_list = []

    for label in class_counts:
        # Extract all samples of a class
        class_samples = [item for item in data_list if item[0] == label]
        # Calculate the number of duplicates needed
        if label == label_up:
            num_needed = class_counts[label]
        else:
            num_needed = 0
        # Shuffle to add randomness
        random.shuffle(class_samples)
        # Append all original samples
        balanced_data_list.extend(class_samples)
        # Append duplicates of the samples randomly until reaching the required count
        for i in range(multiplier,1):
            if num_needed:
                duplicates = class_samples
                balanced_data_list.extend(duplicates)

    # Shuffle the final list to mix classes
    random.shuffle(balanced_data_list)

    return balanced_data_list


def upsampling(data_list):

    class_counts = defaultdict(int)

    # Count the occurrences of each class
    for label, _ in data_list:
        class_counts[label] += 1

    max_samples = max(class_counts.values())

    balanced_data_list = []

    for label in class_counts:
        # Extract all samples of a class
        class_samples = [item for item in data_list if item[0] == label]
        # Calculate the number of duplicates needed
        num_needed = max_samples - class_counts[label]
        # Shuffle to add randomness
        random.shuffle(class_samples)
        # Append all original samples
        balanced_data_list.extend(class_samples)
        # Append duplicates of the samples randomly until reaching the required count
        if num_needed > 0:
            duplicates = random.choices(class_samples, k=num_needed)
            balanced_data_list.extend(duplicates)

    # Shuffle the final list to mix classes
    random.shuffle(balanced_data_list)

    return balanced_data_list
