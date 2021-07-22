import datetime as dt
import numpy as np
from icecream import ic
import pandas as pd
import json
import glob
import Leap_utils as Lp

import os

from datetime import datetime

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from PIL import Image
import cv2

import numpy as np

from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
import logging


def get_epochs(path):
    """
    Returns the Pupil Epoch and System Epoch timestamps

            Parameters:
                    path (str): Path to recording directory
            Returns:
                    timestamps (tuple str): Timestamps read from the file
    """

    f = open(path + "info.player.json")

    pupil_data = json.load(f)  # returns JSON object as a dictionary

    f.close()

    return pupil_data['start_time_synced_s'], pupil_data['start_time_system_s']


def get_leap_timestamps(path):
    """
    Returns the system timestamp and leap timestamp as a dictionary by recording id.

            Parameters:
                    path (str): The path to the patient folder
            Returns:
                    leap_timestamps_dict (datetime obj): A format that allows it to be easily converted to wall time
    """
    leap_timestamps_path = glob.glob(path + r"\*_leap_timestamps.csv")[0]

    with open(leap_timestamps_path) as f:
        leap_data = [x.split(",") for x in f.readlines()]

    leap_timestamps = {}

    for dat in leap_data:
        # Leap timestamp is measured in microseconds
        # System timestamp is the seconds since epoch
        leap_timestamps[dat[0]] = (float(dat[1]) * 10 ** -6, float(dat[2]))

    return leap_timestamps


def get_tone_timestamps(recording_path):

    patient_id = example_path.split("\\")[-1]

    tone_timestamps_path = example_path + "\\" + patient_id + "_tone_timestamps.csv"

    tone_timestamps = {}

    with open(tone_timestamps_path) as f:
        file_data = f.readlines()

    for line in file_data:
        line = line.replace("\n", "")
        line = line.split(',')

        tone_datetime = (datetime.strptime(line[-1], "%Y-%m-%d %H:%M:%S.%f").timestamp())

        tone_timestamps[line[0]] = tone_datetime


    return tone_timestamps


def get_recording_ids(recording_path):
    tone_timestamps = get_tone_timestamps(recording_path)

    return tone_timestamps.keys()


def get_leap_data(recording_path):
    """
    Given the path to the recording folder, find all of the files related to the list
    :return:
    """
    patient_id = example_path.split("\\")[-1]

    leap_list = glob.glob(example_path + r"\*_leap.csv")

    leap_recordings = {}

    for recording_id in leap_list:
        rm_path = recording_id.replace(example_path + "\\", "")
        rm_id = rm_path.replace(patient_id + "_", "")
        leap_recordings[rm_id.replace("_leap.csv", "")] = pd.read_csv(recording_id)

    leap_timestamps = get_leap_timestamps(example_path)

    for key in leap_recordings.keys():
        leap_system, system = leap_timestamps[key]
        ic(system, leap_system)
        leap_df = leap_recordings[key]
        leap_df['timestamp'] *= 10 ** -6
        leap_df['timestamp'] -= leap_system
        leap_df['timestamp'] += system -3600
        ic(leap_df['timestamp'][0])
        ic(leap_recordings[key]['timestamp'][0])

    return leap_recordings


def convert_pupil_timestamp(input_timestamp, pupil_start_time_system, pupil_start_time_synced):
    """
    Returns the converted Pupil Epoch timestamp to Unix Epoch

            Parameters:
                    input_timestamp (int): Time delta measured from Pupil Epoch
                    pupil_start_time_system (int): Unix Epoch at start time of the recording
                    pupil_start_time_synced (int): Pupil Epoch at start time of the recording

            Returns:
                    correlated_timestamp (datetime obj): A format that allows it to be easily converted to wall time
    """
    offset = pupil_start_time_system - pupil_start_time_synced

    return dt.datetime.fromtimestamp(input_timestamp + offset)


def get_world_camera_timestamps(recording_path):

    patient_id = example_path.split("\\")[-1]

    world_timestamps = {}

    for key in get_recording_ids(recording_path):
        pupil_recording_path = example_path + "\\" + patient_id + "_" + key + "\\000\\"
        start_time_synced, start_time_system = ic(get_epochs(pupil_recording_path))

        timestamps = np.load(pupil_recording_path + "world_timestamps.npy")

        adjusted_timestamps = []

        for x in timestamps:
            pupil_timestamp = convert_pupil_timestamp(x, start_time_system, start_time_synced)
            epoch = dt.datetime.utcfromtimestamp(0)
            adjusted_timestamps.append((pupil_timestamp - epoch).total_seconds())

        world_timestamps[key] = np.array(adjusted_timestamps)

    return world_timestamps


def interpolate_leap_for_timestamps(leap_df, timestamps):
    """
    Returns a dataframe of the specified timestamps, by interpolating the recorded data.

            Parameters:
                    leap_df (pandas dataframe): Recorded data.
                    timestamps (array of timestamps): Should be the same format as in leap_df

            Returns:
                    leap_df (pandas dataframe): Interpolated data.

    """

    # Make a copy of the data as we are going to be making changes

    leap_interpolation_df = leap_df.copy(deep=True)

    # 1. Split by hand_id. Each hand_id is a continuous time that the time has been active
    # 2. Find which timestamps occur within each group.


    leap_hand_id_df = leap_interpolation_df.groupby(by="hand_id")

    interpolated_dfs = []

    for name, group in leap_hand_id_df:

        # Creat an blank dataframe, with the same format as the leap data. But for the number of
        # timestamps that we want to interpolate for.

        timestamps_empty = np.empty((len(timestamps), len(leap_df.columns)))
        timestamps_empty[:] = np.nan
        timestamps_df = pd.DataFrame(data=timestamps_empty, columns=leap_df.columns)
        timestamps_df['timestamp'] = timestamps

        # Only keep those timestamps where hands are present in the data.

        min_timestamp = group['timestamp'].min()
        max_timestamp = group['timestamp'].max()

        timestamps_df = timestamps_df[(timestamps_df['timestamp'] >= min_timestamp) &
                                      (timestamps_df['timestamp'] <= max_timestamp)]

        # Concatenate the two dataframes together, and sort it into a format that is ready for
        # interpolation.

        group = pd.concat([group, timestamps_df], axis=0).sort_values('timestamp')

        group = group.set_index('timestamp')  # needed for cubic interpolation

        numeric_group = group.select_dtypes(exclude='object')  # "hand_id" cannot be interpolated as it is a string

        interpolated_group = numeric_group.interpolate(method='cubic', axis=0)  # interpolate across rows

        group = group.fillna(method="ffill")  # forward fills "hand_id"

        group[interpolated_group.columns] = interpolated_group  # Brings in interpolated data

        group = group.reindex(pd.Index(timestamps_df['timestamp']))  # Remove original data

        interpolated_dfs.append(group)  # This is repeated for each instance of a hand

    return pd.concat(interpolated_dfs, axis=0).sort_values('timestamp')


def plot_leap_with_timestamp(fig, data, timestamp):
    """
    Returns a dataframe of the specified timestamps, by interpolating the recorded data.

            Parameters:
                    leap_df (pandas dataframe): Recorded data.
                    timestamps (array of timestamps): Should be the same format as in leap_df

            Returns:
                    leap_df (pandas dataframe): Interpolated data.

    """
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim3d(-300, 150)
    ax.set_ylim3d(-300, 150)
    ax.set_zlim3d(-300, 150)

    ax.view_init(125, -140)

    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    if timestamp in data.index.array:

        tracking_event = data.loc[data.index == timestamp, :]

        tracking_event = Lp.get_tracking_event(tracking_event)  # get correct format for plotting

        for hand in tracking_event.hands:
            Lp.plot_hand(hand, ax)



def make_leap_video(filename, framerate, data, timestamps):
    """
    Creates a video from leap motion data, at specified timestamps.

            Parameters:
                    filename (str): filename, must contain extension.
                    framerate (array of timestamps): frames per second the the video will play
                    data (pandas dataframe: leap motion data)
                    timestamps (array): array of timestamps in unix epoch format.

            Returns:
                    None
    """

    # For each timestamp there is either hand data or there isn't.
    # But for each we need to create a frame anyway.

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, framerate, (500, 400))


    for timestamp in tqdm(timestamps):
        # make a Figure and attach it to a canvas.
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)

        plot_leap_with_timestamp(fig, data, timestamp)

        # Retrieve a view on the renderer buffer
        canvas.draw()
        buf = canvas.buffer_rgba()
        # convert to a NumPy array
        x = np.asarray(buf)

        img = Image.fromarray(x.astype(np.uint8))
        opencvimage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        out.write(opencvimage)
        plt.close(fig)

    out.release()




def file_check(path):

    patient_id = example_path.split("\\")[-1]
    recording_ids = get_recording_ids(path)

    pupil_dir = [path + "\\" + patient_id + "_" + recording_id for recording_id in recording_ids]

    leap_files = glob.glob(path + r"\*_leap.csv")

    report_pictures = glob.glob(path + r"report\*.png")

    # number of recording id's match number of
        # number of pupil core folders?
        # number of report pictures?
        # number of leap files?

    def number_comparator(num_1, num_2, name):
        if num_1 > num_2:
            return "Missing " + str(num_1 - num_2) + "files for " + name + "\n"
        elif num_1 < num_2:
            return "Expected " + str(num_2 - num_1) + "less files for " + name + "\n"
        else:
            return name + "okay"

    output_str = ""

    output_str += number_comparator(len(recording_ids), len(leap_files), "Leap")
    output_str += number_comparator(len(recording_ids), len(report_pictures), "Report")
    output_str += number_comparator(len(recording_ids), len(pupil_dir), "Pupil")

    if os.path.exists(path + "\\" + patient_id + "_demographics.csv"):
        output_str += "No demographics file"
    else:
        output_str += "Demographics file found"


    if os.path.exists(path + "\\" + patient_id + "_tone_timestamps.csv"):
        output_str += "No tone timestamps file"
    else:
        output_str += "Tone timestamps file found"


def check_leap_file_sizes(path):
    leap_files = glob.glob(path + r"\*_leap.csv")

    error_files = []
    for file in leap_files:
        if os.stat(file).st_size/1024 < 1000:
            error_files.append(file)

    return error_files


def hand_visibility_time(ax, title, leap_df, world_timestamps, tone_timestamp):

    leap_copy = leap_df.copy(deep=True)

    min_timestamp = world_timestamps.min() -7200
    max_timestamp = world_timestamps.max() - min_timestamp -7200

    leap_copy['timestamp'] = (leap_copy['timestamp'] - min_timestamp)

    grouped_leap_df = leap_copy.groupby(by=['hand_type', 'hand_id'])

    left = []
    right = []

    for name,group in grouped_leap_df:

       start_dur = (group['timestamp'].min(), group['timestamp'].max() - group['timestamp'].min())

       if 'left' in name:
          left.append(start_dur)
       else:
          right.append(start_dur)


    ax.broken_barh(left, (20, 9), facecolors='black')
    ax.broken_barh(right, (10, 9), facecolors='black')

    plt.ylim((0, 40))

    ymin, ymax = plt.ylim()
    arrowprops = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink': 0.05}


    ax.annotate('Pupil Capture start', xy=(0, ymax-6), xytext=(10, 25), textcoords='offset points',
                rotation=0, va='bottom', ha='left', annotation_clip=True, arrowprops=arrowprops, backgroundcolor="w")

    ax.annotate('Auditory tone', xy=(tone_timestamp- min_timestamp, ymax-9), xytext=(10, 25), textcoords='offset points',
                rotation=0, va='bottom', ha='left', annotation_clip=True, arrowprops=arrowprops, backgroundcolor="w")

    ax.annotate('Pupil Capture stop', xy=(max_timestamp, ymax-6), xytext=(10, 25), textcoords='offset points',
                 rotation=0, va='bottom', ha='left', annotation_clip=True, arrowprops=arrowprops, backgroundcolor="w")

    ax.axvline(tone_timestamp- min_timestamp, alpha=0.5, color='black', label='Auditory tone', dashes=(5, 2, 1, 2))
    ax.axvline(0, alpha=0.5, color='black', label='Pupil start', dashes=(5, 2, 1, 2))
    ax.axvline(max_timestamp, alpha=0.5, color='black', label='Pupil end', dashes=(5, 2, 1, 2))


    plt.title(title)

    ax.set_xlabel('Time ($s$)')
    ax.set_ylabel('Hand')
    ax.set_yticks([15, 25])
    #ax.xlim([-1,30])
    ax.set_xticks(np.arange(0,31, step=2))
    ax.set_yticklabels(['Right', 'Left'])
    ax.grid(False)

    #plt.show()


def plot_leap_velocity(ax, orignial_leap_df, world_timestamps, tone_timestamp, handedness):

    min_timestamp = world_timestamps.min() - 7200
    max_timestamp = world_timestamps.max() - min_timestamp - 7200


    print(world_timestamps.max() - world_timestamps.min())
    print(orignial_leap_df['timestamp'].max() - orignial_leap_df['timestamp'].min())

    leap_df = orignial_leap_df.copy(deep=True)
    leap_df['timestamp'] = (leap_df['timestamp'] - min_timestamp)


    leap_df = leap_df[leap_df['hand_type'] == handedness.lower()]

    grouped_leap_df = leap_df.groupby('hand_id')

    ax.grid(color='#F2F2F2', alpha=1, zorder=0)
    ax.set_xlabel('Time (s)', fontsize=13)

    ax.set_ylabel("Speed ($m$ $s^{-1}$)", fontsize=13)
    #ax.yticks(fontsize=9)
    #ax.xticks(fontsize=9)

    ax.set_title(handedness + "hand speed")

    ax.set_xlim([-0.2, max_timestamp + 1])
    ax.set_xticks(np.arange(0, max_timestamp + 1, step=1))

    for name,group in grouped_leap_df:
        group['velocity_magnitude'] = group[['palm_velocity_x', 'palm_velocity_y', 'palm_velocity_z']] \
            .apply(np.linalg.norm, axis=1)  # euclidean distance

        group['velocity_magnitude'] = savgol_filter(group['velocity_magnitude'], 33, 3)

        ax.plot(group.timestamp, group.velocity_magnitude, label='x direction', color='black')

    ymin, ymax = plt.ylim()

    arrowprops = {'width': 1, 'headwidth': 1, 'headlength': 1, 'shrink': 0.05}

    ax.annotate('Pupil Capture start', xy=(0, ymax * 0.2), xytext=(10, 25), textcoords='offset points',
                rotation=0, va='bottom', ha='left', annotation_clip=True, arrowprops=arrowprops, backgroundcolor="w")

    ax.annotate('Auditory tone', xy=(tone_timestamp- min_timestamp, ymax * 0.1), xytext=(-10, 25), textcoords='offset points',
                rotation=0, va='bottom', ha='right', annotation_clip=True, arrowprops=arrowprops, backgroundcolor="w")

    ax.annotate('Pupil Capture stop', xy=(max_timestamp, ymax * 0.2), xytext=(10, 25), textcoords='offset points',
                 rotation=0, va='bottom', ha='left', annotation_clip=True, arrowprops=arrowprops, backgroundcolor="w")

    ax.axvline(tone_timestamp- min_timestamp, alpha=0.5, color='black', label='Auditory tone', dashes=(5, 2, 1, 2))
    ax.axvline(0, alpha=0.5, color='black', label='Pupil start', dashes=(5, 2, 1, 2))
    ax.axvline(max_timestamp, alpha=0.5, color='black', label='Pupil end', dashes=(5, 2, 1, 2))

    ax.axvline(tone_timestamp - min_timestamp, alpha=0.5, color='black', label='Auditory tone', dashes=(5, 2, 1, 2))


def create_plot():

    fig, axes = plt.subplots(1, 2)


    ax_1 = axes[0][0]
    plot_leap_velocity(ax_1, leap_df[key], world_timestamps[key], tone_timestamps[key], patient_info['handedness'])

    ax_2 = axes[0][1]
    plot_leap_velocity(ax_2, leap_df[key], world_timestamps[key], tone_timestamps[key], patient_info['handedness'])

    fig.suptitle('Main title')
    fig.tight_layout()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    ic.disable()

    example_path = r"E:\Ruijin\Recording\20210621-132108_songshanying"

    leap_df = get_leap_data(example_path)
    world_timestamps = get_world_camera_timestamps(example_path)
    tone_timestamps = get_tone_timestamps(example_path)

    key = list(leap_df.keys())[0]

    patient_id = example_path.split("\\")[-1]

    patient_info = {}

    with open(example_path + "\\" + patient_id + "_intro.csv") as f:
        for line in f.readlines():
            line = line.replace("\n","").split(",")
            patient_info[line[0]] = line[-1]


    fig, axes = plt.subplots(2, 1)
    fig.set_size_inches(15, 12, forward=True)

    ax_1 = axes[0]
    plot_leap_velocity(ax_1, leap_df[key], world_timestamps[key], tone_timestamps[key], patient_info['handedness'])

    ax_2 = axes[1]
    hand_visibility_time(ax_2, "Hand visibility" , leap_df[key], world_timestamps[key], tone_timestamps[key])

    fig.suptitle(key, fontweight='bold', fontsize=24)

    plt.show()

    # plot_leap_velocity(patient_info['handedness'] + " hand speed\n", "\n" + patient_id + " " + key, leap_df[key], world_timestamps[key], tone_timestamps[key], patient_info['handedness'])
    #
    # hand_visibility_time("Hand visibility\n" +patient_id + " " + key, leap_df[key], world_timestamps[key], tone_timestamps[key])


    # check_leap_file_sizes(example_path)
    #
    # # ic.enable()
    #
    # # tone_timestamps
    # tone_timestamps = "E:\Ruijin\Recording\20210621-132108_songshanying\20210621-132108_songshanying_tone_timestamps.csv"
    #
    # leap_recordings_keys = list(leap_recordings)
    # example_key = leap_recordings_keys[0]
    #
    # data = interpolate_leap_for_timestamps(leap_recordings[example_key], world_timestamps[example_key])
    #
    # make_leap_video("test.mp4", 60.00, data, world_timestamps[example_key])

# Timestamps are only guaranteed to be unique in the case of hand id
# For each timestamp we need to know if it lies between the start and finish of a hand id.
# For each hand id, we have timestamps that we want to sample from

# for key in leap_recordings.keys():
#     leap_video_df = leap_recordings[key].copy(deep=True)
#     grouped_leap_df = leap_video_df.groupby(by=['hand_id'])
#
#     world_timestamp = world_timestamps[key]
#     world_timestamp_index = pd.Float64Index(world_timestamp)
#
#     for name, group in grouped_leap_df:
#         world_timestamp = world_timestamps[key]
#
#
#
#         #leap_timestamps = group.index
#
#         #joint_series = leap_timestamps.append(pd.Float64Index(world_timestamp)).drop_duplicates().sort_values()
#
#         #ic(joint_series)
#
#         #group.to_csv(name + key + ".csv")
#         #ic(joint_series.shape)
#         #ic(joint_series.nunique())
#     #ic(leap_video_df['timestamp'])
#     #test change ddd
#
#
#
#         #group = group.reindex(joint_series)
#
#         #group.to_csv(key + ".csv")
#
#     break
# file_list = ic(leap_recordings.keys()) # Files in the folder

# 1. have all of the relevant files available
# 2. Apply the pre-processing steps so that the data can be used together
