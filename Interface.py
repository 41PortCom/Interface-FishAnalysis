# Fish Tracking and Analysis Code
# Copyright (c) 2024 [I.E.T.R]
# This code is licensed under the MIT License (see LICENSE for details).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
from nptdms import TdmsFile
from scipy.signal import butter, filtfilt
from collections import deque

# --- File selection dialogs ---
Tk().withdraw()

csv_file = askopenfilename(title="Select the CSV file", filetypes=[("CSV Files", "*.csv")])
if not csv_file:
    print("No CSV file selected. Exiting program.")
    exit()

data = pd.read_csv(csv_file, header=None)
if data.empty or data.shape[1] < 3:
    print("The CSV file is empty or incorrectly formatted.")
    exit()

mp4_file = askopenfilename(title="Select the video file (.mp4)", filetypes=[("MP4 Files", "*.mp4")])
if not mp4_file:
    print("No video file selected. Exiting program.")
    exit()

tdms_file = askopenfilename(title="Select the TDMS file", filetypes=[("TDMS Files", "*.tdms")])
if not tdms_file:
    print("No TDMS file selected. Exiting program.")
    exit()

# --- Signal Processing Functions ---
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.3, highcut=15, fs=500, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Read TDMS data
tdms_data = TdmsFile.read(tdms_file)
channels_dict = {channel.name: channel for group in tdms_data.groups() for channel in group.channels()}

pressure_data = {}
channels_to_read = ["S1", "S2", "S3", "S4", "S6", "S7"]

for channel_name in channels_to_read:
    if channel_name in channels_dict:
        raw_data = channels_dict[channel_name][:]
        pressure_data[channel_name] = bandpass_filter(raw_data)
    else:
        print(f"Channel {channel_name} not found in the file.")

# --- CSV Data Extraction ---
timestamps = data.iloc[:, 0] / 4
X = data.iloc[:, 1::2].values * 0.4
Y = data.iloc[:, 2::2].values * 0.4

# --- Video Setup ---
cap = cv2.VideoCapture(mp4_file)
fps = cap.get(cv2.CAP_PROP_FPS) * 4
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps

# --- Plot Setup ---
fig = plt.figure(num="FISH ANALYSIS", figsize=(16, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1], width_ratios=[3, 2], hspace=0.5)
ax_video = fig.add_subplot(gs[0, 0])
ax_info = fig.add_subplot(gs[0, 1])
ax_pressure = fig.add_subplot(gs[1, :])

ret, frame = cap.read()
if not ret:
    print("Unable to read video.")
    exit()

img_display = ax_video.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
scat = ax_video.scatter(X[0, :], Y[0, :], s=1, color='blue')
spine_point, = ax_video.plot([], [], 'ro', markersize=4)

ax_video.set_xlim(0, width)
ax_video.set_ylim(height, 0)
ax_video.set_xlabel('X Position')
ax_video.set_ylabel('Y Position')
ax_video.set_title('Video and Fish Spine')

wall1_y, wall2_y = 145, 300
ax_video.plot([0, width], [wall1_y, wall1_y], 'g-', linewidth=2)
ax_video.plot([0, width], [wall2_y, wall2_y], 'g-', linewidth=2)

sensor_positions = {
    "sensor1 (mm)": np.array([620, 300]),
    "sensor2 (mm)": np.array([465, 300]),
    "sensor3 (mm)": np.array([300, 300]),
    "sensor4 (mm)": np.array([620, 145]),
    "sensor5 (mm)": np.array([465, 145]),
    "sensor6 (mm)": np.array([300, 145]),
}

for pos in sensor_positions.values():
    ax_video.plot(pos[0], pos[1], 'ro', markersize=5)

table_data = [
    ['Sensors', 'Data'],
    *[[sensor, ''] for sensor in sensor_positions.keys()],
    ['Speed(mm/sec)', ''],
    ['Acceleration(mm/sec²)', ''],
    ['Distance Wall 1 (mm)', ''],
    ['Distance Wall 2 (mm)', ''],
    ['Pressure (Pa)', ''],
    ['Angle (deg)', '']
]
table_info = ax_info.table(cellText=table_data, cellLoc='center', loc='center')
ax_info.axis('off')

ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (s)', 0, duration, valinit=0, valstep=10/fps)

ax_radio = plt.axes([0.05, 0.15, 0.05, 0.15])
radio = RadioButtons(ax_radio, list(pressure_data.keys()))

# --- Metric Calculations ---
def calculate_distance_to_sensor(fish_position, sensor_position):
    return np.linalg.norm(fish_position - sensor_position) * 0.5145 / 0.4

def calculate_speed(previous_position, current_position, time_delta):
    if time_delta > 0:
        return np.linalg.norm(current_position - previous_position) / time_delta * 0.5145 / 0.4
    return 0

def calculate_distance_to_wall(fish_y, wall_y):
    return abs(fish_y - wall_y) * 0.5145 / 0.4

distances_to_sensors_all = {sensor: np.zeros(len(timestamps)) for sensor in sensor_positions.keys()}
distance_to_wall1_all = np.zeros(len(timestamps))
distance_to_wall2_all = np.zeros(len(timestamps))
speed_all = np.zeros(len(timestamps))
sliding_window_speed = deque(maxlen=5)

for i in range(1, len(timestamps)):
    current_position = np.array([X[i, 35], Y[i, 35]])
    previous_position = np.array([X[i - 1, 35], Y[i - 1, 35]])

    for sensor, pos in sensor_positions.items():
        distances_to_sensors_all[sensor][i] = calculate_distance_to_sensor(current_position, pos)

    distance_to_wall1_all[i] = calculate_distance_to_wall(current_position[1], wall1_y)
    distance_to_wall2_all[i] = calculate_distance_to_wall(current_position[1], wall2_y)

    time_delta = timestamps.iloc[i] - timestamps.iloc[i - 1]
    spd = calculate_speed(previous_position, current_position, time_delta)
    sliding_window_speed.append(spd)
    speed_all[i] = np.mean(sliding_window_speed)

acceleration_all = np.zeros(len(timestamps))
sliding_window_acc = deque(maxlen=5)
for i in range(1, len(timestamps)):
    dt = timestamps.iloc[i] - timestamps.iloc[i - 1]
    acc = (speed_all[i] - speed_all[i - 1]) / dt if dt > 0 else 0
    sliding_window_acc.append(acc)
    acceleration_all[i] = np.mean(sliding_window_acc)

output_filename = "output.txt"
with open(output_filename, "w") as f:
    f.write("Timestamp\tSpeed(mm/sec)\tNearestSensor(mm)\tNearestWall(mm)\tAcceleration(mm/sec^2)\tPressure (Pa)\tAngle (deg)\n")
    working_sensors = ['sensor1 (mm)', 'sensor2 (mm)', 'sensor4 (mm)']
    sensor_to_channel = {"sensor1 (mm)": "S1", "sensor2 (mm)": "S2", "sensor4 (mm)": "S4"}
    previous_sensor = None

    for i in range(len(timestamps)):
        ts = timestamps.iloc[i]
        spd = speed_all[i]
        nearest_sensor_name = min(working_sensors, key=lambda s: distances_to_sensors_all[s][i])

        if previous_sensor is not None and nearest_sensor_name != previous_sensor:
            f.write("0.000\t162.636\t711.780\t186.506\t1128.613\t0.066\t32.829\n")
        previous_sensor = nearest_sensor_name

        nearest_sensor_distance = distances_to_sensors_all[nearest_sensor_name][i]
        nearest_wall = min(distance_to_wall1_all[i], distance_to_wall2_all[i])
        acc = acceleration_all[i]
        channel_pressure = sensor_to_channel[nearest_sensor_name]
        sample_index = int(ts * 500)
        pressure_value = pressure_data[channel_pressure][sample_index] if sample_index < len(pressure_data[channel_pressure]) else float('nan')

        fish_position = np.array([X[i, 35], Y[i, 35]])
        sensor_position = sensor_positions[nearest_sensor_name]
        delta_x = fish_position[0] - sensor_position[0]
        delta_y = fish_position[1] - sensor_position[1]
        angle_rad = np.arctan2(delta_x, delta_y)
        angle_deg = np.degrees(angle_rad)

        f.write(f"{ts:.3f}\t{spd:.3f}\t{nearest_sensor_distance:.3f}\t{nearest_wall:.3f}\t{acc:.3f}\t{pressure_value:.3f}\t{angle_deg:.3f}\n")

print(f"Data saved to '{output_filename}'.")

# --- Update Function ---
def update(val):
    time_in_seconds = time_slider.val
    frame_idx = np.argmin(np.abs(timestamps - time_in_seconds))

    cap.set(cv2.CAP_PROP_POS_MSEC, time_in_seconds * 1000 * 4)
    ret, frame = cap.read()
    if ret:
        img_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    scat.set_offsets(np.c_[X[frame_idx, :], Y[frame_idx, :]])
    spine_point.set_data([X[frame_idx, 35]], [Y[frame_idx, 35]])

    for i, sensor in enumerate(sensor_positions.keys()):
        table_info[i + 1, 1].get_text().set_text(f'{distances_to_sensors_all[sensor][frame_idx]:.2f}')
    table_info[len(sensor_positions) + 1, 1].get_text().set_text(f'{speed_all[frame_idx]:.2f}')
    table_info[len(sensor_positions) + 2, 1].get_text().set_text(f'{acceleration_all[frame_idx]:.2f}')
    table_info[len(sensor_positions) + 3, 1].get_text().set_text(f'{distance_to_wall1_all[frame_idx]:.2f}')
    table_info[len(sensor_positions) + 4, 1].get_text().set_text(f'{distance_to_wall2_all[frame_idx]:.2f}')

    selected_channel = radio.value_selected
    if selected_channel:
        sample_in_index = int(time_in_seconds * 500)
        half_window = int(10 * 500)
        start_idx = max(0, sample_in_index - half_window)
        end_idx = min(len(pressure_data[selected_channel]), sample_in_index + half_window)

        table_info[len(sensor_positions) + 5, 1].get_text().set_text(f'{pressure_data[selected_channel][sample_in_index]:.2f}')
        ax_pressure.clear()
        time_axis = np.linspace(time_in_seconds - 5, time_in_seconds + 5, end_idx - start_idx)
        ax_pressure.plot(time_axis, pressure_data[selected_channel][start_idx:end_idx])
        ax_pressure.axvline(x=time_in_seconds, color='red', linestyle='--', linewidth=1)
        ax_pressure.set_ylim(-1.5, 1)
        ax_pressure.set_xlim(time_in_seconds - 5, time_in_seconds + 5)
        ax_pressure.set_title(f'Pressure Data for {selected_channel}')
        ax_pressure.set_xlabel('Time (s)')
        ax_pressure.set_ylabel('Pressure (Pa)')

    nearest_sensor_name = min(['sensor1 (mm)', 'sensor2 (mm)', 'sensor4 (mm)'], key=lambda s: distances_to_sensors_all[s][frame_idx])
    fish_position = np.array([X[frame_idx, 35], Y[frame_idx, 35]])
    sensor_position = sensor_positions[nearest_sensor_name]
    angle_deg = np.degrees(np.arctan2(fish_position[0] - sensor_position[0], fish_position[1] - sensor_position[1]))
    table_info[len(sensor_positions) + 6, 1].get_text().set_text(f'{angle_deg:.2f}')

    fig.canvas.draw_idle()

time_slider.on_changed(update)

def on_key(event):
    if event.key == 'right':
        current_time = time_slider.val + (10 / fps)
        time_slider.set_val(min(current_time, duration))
    elif event.key == 'left':
        current_time = time_slider.val - (10 / fps)
        time_slider.set_val(max(current_time, 0))
    update(time_slider.val)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
cap.release()
cv2.destroyAllWindows()
