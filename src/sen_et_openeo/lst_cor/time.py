import os
import re
from datetime import datetime, timedelta


def extract_datetime_strings_from_filename(filename):
    filename = os.path.basename(filename)

    # Define the pattern for matching datetime strings
    pattern = r'(\d{8}T\d{6})'

    # Use regular expression to find matches in the filename
    matches = re.findall(pattern, filename)

    # Check if at least two datetime strings were found
    if len(matches) >= 2:
        # Extract the first two datetime strings
        timestring1, timestring2 = matches[:2]
        return timestring1, timestring2
    else:
        # Handle the case when not enough datetime strings are found
        return None, None


def time_differences_two_yyyymmddthhmmss_strings(str1, str2):
    datetime1 = get_datetime_from_yyyymmddthhmmss(str1)
    datetime2 = get_datetime_from_yyyymmddthhmmss(str2)
    time_difference = datetime1 - datetime2

    return time_difference


def get_datetime_from_yyyymmddthhmmss(yyyymmddthhmmss):
    date_format = "%Y%m%dT%H%M%S"
    datetime_format = datetime.strptime(yyyymmddthhmmss, date_format)

    return datetime_format


def find_simult_acquisitions(target_time, folder):
    sub_folders = os.listdir(folder)
    acquisition_times = [get_datetime_from_yyyymmddthhmmss(
        sub_folder) for sub_folder in sub_folders]

    # Filter subfolder names based on whether their acquisition times are less than 10 minutes away from the target time
    result_folders = [sub_folder for sub_folder, acq_time in zip(sub_folders, acquisition_times)
                      if abs(acq_time - target_time) < timedelta(minutes=10)]

    return result_folders
