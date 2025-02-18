import re
from collections import defaultdict
import argparse

def analyze_log(file_path):
    print(f"Analyzing log file: {file_path}")
    lane_actions = ['LANE_LEFT', 'LANE_RIGHT']
    other_actions = ['FASTER', 'SLOWER', 'IDLE']

    counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    run_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Regular expression to match action lines and capture Time, action, discard_reason, and status
    pattern = re.compile(
        r'\[SafetyShield\] Time (\d+), action (\w+)(?: discarded due to (\w+))?(?:\s+(\w+))?\.'
    )

    current_test_run = None
    processed_times = set()
    total_processed_actions = 0  # To track total actions processed across all runs

    try:
        with open(file_path, 'r') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if line.startswith('Test run'):
                    current_test_run = line
                    processed_times = set()
                    continue
                match = pattern.search(line)
                if match and current_test_run:
                    time = int(match.group(1))
                    action = match.group(2)
                    discard_reason = match.group(3)
                    action_status = match.group(4)

                    if time not in processed_times:
                        processed_times.add(time)
                        total_processed_actions += 1

                        if discard_reason == 'unsafe':
                            status = 'discarded_due_to_unsafe'
                        elif discard_reason == 'unavailable':
                            status = 'discarded_due_to_unavailable'
                        elif action_status == 'passed':
                            status = 'passed'
                        elif action_status == 'ignored':
                            status = 'ignored'
                        else:
                            status = 'unknown'

                        if status != 'unknown':
                            counts[action][status] += 1
                            total_counts[status] += 1
                            run_counts[current_test_run][action][status] += 1
                        else:
                            print(f"Line {line_number}: Unrecognized status in line: {line}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return

    print(f"\nTotal actions: {total_processed_actions}\n")

    print("Lane Change Actions:")
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
