import re
from collections import defaultdict
import argparse

def analyze_log(file_path):
    print(f"Analyzing log file: {file_path}")
    lane_actions = ['LANE_LEFT', 'LANE_RIGHT']
    other_actions = ['FASTER', 'SLOWER', 'IDLE']

    counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)

    # Regular expression to match action lines
    pattern = re.compile(r'action (\w+)\s*(?:discarded due to (\w+))?(?:\s*ignored)?(?:\s*passed)?\.')

    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                match = pattern.search(line)
                if match:
                    action = match.group(1)
                    status = match.group(2)
                    # print(f"Line {line_number}: Matched action '{action}', status '{status}'")

                    if 'passed' in line:
                        counts[action]['passed'] += 1
                        total_counts['passed'] += 1
                    elif 'ignored' in line:
                        counts[action]['ignored'] += 1
                        total_counts['ignored'] += 1
                    elif status == 'unsafe':
                        counts[action]['discarded_due_to_unsafe'] += 1
                        total_counts['discarded_due_to_unsafe'] += 1
                    elif status == 'unavailable':
                        counts[action]['discarded_due_to_unavailable'] += 1
                        total_counts['discarded_due_to_unavailable'] += 1
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    total_actions = sum(total_counts.values())
    print(f"Total actions: {total_actions}")

    print("\nLane Change Actions:")
    for action in lane_actions:
        print(f"\nAction: {action}")
        for status in ['passed', 'discarded_due_to_unsafe', 'discarded_due_to_unavailable', 'ignored']:
            count = counts[action].get(status, 0)
            print(f"  {status}: {count}")

    print("\nOther Actions:")
    for action in other_actions:
        print(f"\nAction: {action}")
        for status in ['passed', 'discarded_due_to_unsafe', 'discarded_due_to_unavailable', 'ignored']:
            count = counts[action].get(status, 0)
            print(f"  {status}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze log file.')
    parser.add_argument('log_file', help='Path to the log file')
    args = parser.parse_args()
    analyze_log(args.log_file)
