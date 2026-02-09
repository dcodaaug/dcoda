import re

# Custom sort function to extract the numeric part and sort based on it
def sort_key(s):
    # Extract the digits from the string
    match = re.search(r'\d+', s)
    return int(match.group()) if match else float('inf')
