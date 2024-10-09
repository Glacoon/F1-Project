import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlencode
from urllib.request import urlopen
from datetime import timedelta
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colormaps

from PIL import Image

# Load the car image
try:
    car_img = Image.open('redbull_pixel_art_noBG.png')
    car_img = car_img.rotate(90)  # Rotate the image if needed
    car_img = car_img.resize((20, 20))  # Resize the image as needed
    car_img = np.array(car_img)
    print("Car image loaded successfully. Shape:", car_img.shape)
except FileNotFoundError:
    print("Car image file not found. Using a placeholder.")
    car_img = np.ones((20, 20, 4))  # Create a simple placeholder
    print("Placeholder image created. Shape:", car_img.shape)

# List of F1 countries and USA circuits
country_list = ['Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'United States', 'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Singapore', 'Japan', 'Qatar', 'Mexico', 'Brazil', 'Abu Dhabi']
usa_circuits = ['Miami', 'Austin', 'Las Vegas']

# Select the country
country = input('Enter the country: ')
while country not in country_list:
    country = input('Invalid country. Please enter a valid country: ')

# Handle the case of the United States to choose from available circuits
if country == 'United States':
    circuit = input('Enter the city (Miami, Austin, Las Vegas): ')
    while circuit not in usa_circuits:
        circuit = input('Invalid city. Please enter a valid city (Miami, Austin, Las Vegas): ')

# Ask for session type using a selection
session_type_list = ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race', 'Sprint', 'Sprint Shootout']
session_type = input('Enter the session type: ')
while session_type not in session_type_list:
    session_type = input('Invalid session type. Please enter a valid session type: ')

# Prepare URL parameters
params = {
    'country_name': country,
    'session_name': session_type
}
if country == 'United States':
    params['location'] = circuit

# Encode URL parameters
url = f'https://api.openf1.org/v1/sessions?{urlencode(params)}'

response = urlopen(url)
session_data = json.loads(response.read().decode('utf-8'))

# Check if session data is not empty
if not session_data:
    raise ValueError("No session data found for the specified criteria.")

# Ask which driver and lap number
driver_number = input('Enter driver number: ')
lap_number = input('Enter lap number: ')

# Fetch lap data for the specific driver and lap
response_lap = urlopen(f'https://api.openf1.org/v1/laps?session_key={session_data[0]["session_key"]}&driver_number={driver_number}&lap_number={lap_number}')
lap_data = json.loads(response_lap.read().decode('utf-8'))

# Ensure lap_data is a list
if isinstance(lap_data, list) and lap_data:
    lap_data = lap_data[0]  # Access the first element if it's a list
else:
    raise ValueError("Invalid lap data format.")

# Extract date_start and calculate date_end (date_start + lap_duration + 3 seconds)
date_start = pd.to_datetime(lap_data['date_start'])
lap_duration = timedelta(seconds=lap_data['lap_duration'])
date_end = date_start + lap_duration + timedelta(seconds=3)

print(f"Lap start: {date_start}, Lap end (+3 sec): {date_end}")

# Fetch location data for the specific lap using the time range
# Encode the dates to avoid control characters in the URL
date_start_str = date_start.isoformat()
date_end_str = date_end.isoformat()
params_location = {
    'session_key': session_data[0]["session_key"],
    'driver_number': driver_number,
    'date>': date_start_str,
    'date<': date_end_str
}
url_location = f'https://api.openf1.org/v1/location?{urlencode(params_location)}'
response_location = urlopen(url_location)
location_data = json.loads(response_location.read().decode('utf-8'))

# Check if location data is not empty
if not location_data:
    raise ValueError("No location data found for the specified driver and lap.")

df_location = pd.DataFrame(location_data)

# Fetch car data for speed
response_car_data = urlopen(f'https://api.openf1.org/v1/car_data?driver_number={driver_number}&session_key={session_data[0]["session_key"]}&speed>=0')
car_data = json.loads(response_car_data.read().decode('utf-8'))

# Check if car data is not empty
if not car_data:
    raise ValueError("No car data found for the specified driver.")

df_car_data = pd.DataFrame(car_data)

# Convert 'date' columns to datetime, handling mixed formats
df_location['date'] = pd.to_datetime(df_location['date'], format='mixed')
df_car_data['date'] = pd.to_datetime(df_car_data['date'], format='mixed')

# Merge the location data with car data on 'date' or 'timestamp'
df_merged = pd.merge_asof(df_location.sort_values('date'), df_car_data.sort_values('date'), on='date', direction='nearest')

# Calculate 'distance' if missing
if 'distance' not in df_merged.columns:
    df_merged['distance'] = np.sqrt((df_merged['x'].diff()**2 + df_merged['y'].diff()**2).cumsum())
    df_merged['distance'] = df_merged['distance'].fillna(0)

# Normalize speed for color mapping (min speed = 0, max speed = 315 for F1 cars)
norm = plt.Normalize(vmin=df_merged['speed'].min(), vmax=df_merged['speed'].max())

# Use the new method to get the colormap
cmap = colormaps.get_cmap('jet')

# Create figure and axis
fig, ax = plt.subplots(figsize=(25, 15))

# Calculate padding
padding_x = (df_merged['x'].max() - df_merged['x'].min()) * 0.05
padding_y = (df_merged['y'].max() - df_merged['y'].min()) * 0.05

# Set up the plot limits with padding
ax.set_xlim(df_merged['x'].min() - padding_x, df_merged['x'].max() + padding_x)
ax.set_ylim(df_merged['y'].min() - padding_y, df_merged['y'].max() + padding_y)

# Create a color bar to indicate speed
norm = plt.Normalize(vmin=df_merged['speed'].min(), vmax=df_merged['speed'].max())
cmap = colormaps.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Speed (km/h)")

ax.set_title('Track with speed-indicated segments (heatmap)')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Initialize the list to store the line segments
lines = []

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlencode
from urllib.request import urlopen
from datetime import timedelta
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colormaps

from PIL import Image

# Load the car image
try:
    car_img = Image.open('redbull_pixel_art_noBG.png')
    car_img = car_img.rotate(90)  # Rotate the image if needed
    car_img = car_img.resize((20, 20))  # Resize the image as needed
    car_img = np.array(car_img)
except FileNotFoundError:
    print("Car image file not found. Using a placeholder.")
    car_img = np.ones((20, 20, 4))  # Create a simple placeholder

# List of F1 countries and USA circuits
country_list = ['Bahrain', 'Saudi Arabia', 'Australia', 'Azerbaijan', 'United States', 'Monaco', 'Spain', 'Canada', 'Austria', 'Great Britain', 'Hungary', 'Belgium', 'Netherlands', 'Italy', 'Singapore', 'Japan', 'Qatar', 'Mexico', 'Brazil', 'Abu Dhabi']
usa_circuits = ['Miami', 'Austin', 'Las Vegas']

# Select the country
country = input('Enter the country: ')
while country not in country_list:
    country = input('Invalid country. Please enter a valid country: ')

# Handle the case of the United States to choose from available circuits
if country == 'United States':
    circuit = input('Enter the city (Miami, Austin, Las Vegas): ')
    while circuit not in usa_circuits:
        circuit = input('Invalid city. Please enter a valid city (Miami, Austin, Las Vegas): ')

# Ask for session type using a selection
session_type_list = ['Practice 1', 'Practice 2', 'Practice 3', 'Qualifying', 'Race', 'Sprint', 'Sprint Shootout']
session_type = input('Enter the session type: ')
while session_type not in session_type_list:
    session_type = input('Invalid session type. Please enter a valid session type: ')

# Prepare URL parameters
params = {
    'country_name': country,
    'session_name': session_type
}
if country == 'United States':
    params['location'] = circuit

# Encode URL parameters
url = f'https://api.openf1.org/v1/sessions?{urlencode(params)}'

response = urlopen(url)
session_data = json.loads(response.read().decode('utf-8'))

# Check if session data is not empty
if not session_data:
    raise ValueError("No session data found for the specified criteria.")

# Ask which driver and lap number
driver_number = input('Enter driver number: ')
lap_number = input('Enter lap number: ')

# Fetch lap data for the specific driver and lap
response_lap = urlopen(f'https://api.openf1.org/v1/laps?session_key={session_data[0]["session_key"]}&driver_number={driver_number}&lap_number={lap_number}')
lap_data = json.loads(response_lap.read().decode('utf-8'))

# Ensure lap_data is a list
if isinstance(lap_data, list) and lap_data:
    lap_data = lap_data[0]  # Access the first element if it's a list
else:
    raise ValueError("Invalid lap data format.")

# Extract date_start and calculate date_end (date_start + lap_duration + 3 seconds)
date_start = pd.to_datetime(lap_data['date_start'])
lap_duration = timedelta(seconds=lap_data['lap_duration'])
date_end = date_start + lap_duration + timedelta(seconds=3)

print(f"Lap start: {date_start}, Lap end (+3 sec): {date_end}")

# Fetch location data for the specific lap using the time range
# Encode the dates to avoid control characters in the URL
date_start_str = date_start.isoformat()
date_end_str = date_end.isoformat()
params_location = {
    'session_key': session_data[0]["session_key"],
    'driver_number': driver_number,
    'date>': date_start_str,
    'date<': date_end_str
}
url_location = f'https://api.openf1.org/v1/location?{urlencode(params_location)}'
response_location = urlopen(url_location)
location_data = json.loads(response_location.read().decode('utf-8'))

# Check if location data is not empty
if not location_data:
    raise ValueError("No location data found for the specified driver and lap.")

df_location = pd.DataFrame(location_data)

# Fetch car data for speed
response_car_data = urlopen(f'https://api.openf1.org/v1/car_data?driver_number={driver_number}&session_key={session_data[0]["session_key"]}&speed>=0')
car_data = json.loads(response_car_data.read().decode('utf-8'))

# Check if car data is not empty
if not car_data:
    raise ValueError("No car data found for the specified driver.")

df_car_data = pd.DataFrame(car_data)

# Convert 'date' columns to datetime, handling mixed formats
df_location['date'] = pd.to_datetime(df_location['date'], format='mixed')
df_car_data['date'] = pd.to_datetime(df_car_data['date'], format='mixed')

# Merge the location data with car data on 'date' or 'timestamp'
df_merged = pd.merge_asof(df_location.sort_values('date'), df_car_data.sort_values('date'), on='date', direction='nearest')

# Calculate 'distance' if missing
if 'distance' not in df_merged.columns:
    df_merged['distance'] = np.sqrt((df_merged['x'].diff()**2 + df_merged['y'].diff()**2).cumsum())
    df_merged['distance'] = df_merged['distance'].fillna(0)

# Normalize speed for color mapping (min speed = 0, max speed = 315 for F1 cars)
norm = plt.Normalize(vmin=df_merged['speed'].min(), vmax=df_merged['speed'].max())

# Use the new method to get the colormap
cmap = colormaps.get_cmap('jet')

# Create figure and axis
fig, ax = plt.subplots(figsize=(25, 15))

# Calculate padding
padding_x = (df_merged['x'].max() - df_merged['x'].min()) * 0.05
padding_y = (df_merged['y'].max() - df_merged['y'].min()) * 0.05

# Set up the plot limits with padding
ax.set_xlim(df_merged['x'].min() - padding_x, df_merged['x'].max() + padding_x)
ax.set_ylim(df_merged['y'].min() - padding_y, df_merged['y'].max() + padding_y)

# Create a color bar to indicate speed
norm = plt.Normalize(vmin=df_merged['speed'].min(), vmax=df_merged['speed'].max())
cmap = colormaps.get_cmap('jet')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="Speed (km/h)")

ax.set_title('Track with speed-indicated segments (heatmap)')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Initialize the list to store the line segments
lines = []

def update(frame):
    global lines
    if frame < len(df_merged) - 1:
        # Extract the current and next coordinates
        x_values = [df_merged.iloc[frame]['x'], df_merged.iloc[frame+1]['x']]
        y_values = [df_merged.iloc[frame]['y'], df_merged.iloc[frame+1]['y']]

        # Get the color based on the speed at this frame
        color = cmap(norm(df_merged.iloc[frame]['speed']))

        # Plot the current track segment
        line, = ax.plot(x_values, y_values, color=color, lw=2, zorder=2)
        lines.append(line)

        # Remove old car image
        if hasattr(update, 'car_image'):
            update.car_image.remove()

        # Add the car image at the current position
        car_x, car_y = df_merged.iloc[frame]['x'], df_merged.iloc[frame]['y']
        update.car_image = ax.imshow(car_img, extent=[car_x-0.5, car_x+0.5, car_y-0.5, car_y+0.5], zorder=3)

    # Remove old lines to prevent memory issues
    if len(lines) > 100:
        old_line = lines.pop(0)
        old_line.remove()

    return lines + [update.car_image]

# Create the animation and store it in a variable
anim = FuncAnimation(fig, update, frames=len(df_merged)-1, blit=True, interval=15, cache_frame_data=False)

# Save the animation as a GIF file
anim.save('f1_animation_with_car.gif', writer=PillowWriter(fps=30))

# Show the plot
plt.show()

# Create the animation and store it in a variable
anim = FuncAnimation(fig, update, frames=len(df_merged)-1, blit=True, interval=15)

# Save the animation as a GIF file
anim.save('f1_animation_with_car.gif', writer=PillowWriter(fps=30))

# Show the plot
plt.show()