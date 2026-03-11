"""
Data Generation Script for Deep Learning Assignment 1

This script generates:
1. A base dataset of 500 samples collected over 15 days
2. An extended dataset of 5000 samples by adding Gaussian noise

The dataset simulates IIT-H campus mess behavior semi-realistically,
based on temporal, behavioral, and environmental factors.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime

# -------------------------------
# Reproducibility
# -------------------------------

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------
# Global Parameters
# -------------------------------

NUM_SAMPLES = 500
NUM_STUDENTS = 150
NUM_DAYS = 15

# -------------------------------
# Student Population Generation
# -------------------------------

def generate_students(num_students):
    """
    Generates synthetic student population.
    Year distribution biased toward 2nd and 3rd year students, as I would be more likely to interact with them as a 2nd year myself:
    1st: 15%
    2nd: 45%
    3rd: 35%
    4th: 05%
    """

    year_probs = [0.15, 0.45, 0.35, 0.05]
    years = np.random.choice([1, 2, 3, 4],
                             size=num_students,
                             p=year_probs)

    student_ids = np.arange(num_students)

    return student_ids, years

# -------------------------------
# Meal Time Windows (in minutes from midnight)
# -------------------------------

MEAL_WINDOWS = {
    0: (7*60 + 30, 10*60),       # Breakfast: 7:30 - 10:00
    1: (12*60 + 30, 14*60 + 30), # Lunch: 12:30 - 14:30
    2: (19*60 + 30, 21*60 + 30)  # Dinner: 19:30 - 21:30
}

def sample_arrival_time(meal_type, day_of_week):
    """
    Sample arrival time (in minutes from midnight, as people are likely to report times by rounding to the nearest minute).
    Includes rush peaks caused by class times.
    """

    start, end = MEAL_WINDOWS[meal_type]

    # Sample uniformly first
    t = np.random.randint(start, end + 1)

    # Define rush peaks in minutes
    if meal_type == 0:  # Breakfast
        peaks = [8*60 + 45, 9*60 + 45]  # 8:45, 9:45

        for peak in peaks:
            if abs(t - peak) < 10 and np.random.rand() < 0.6:
                t = int(np.random.normal(peak, 5))

    elif meal_type == 1:  # Lunch
        peaks = [13*60 + 15, 14*60]  # 13:15, 14:00

        for peak in peaks:
            if abs(t - peak) < 10 and np.random.rand() < 0.5:
                t = int(np.random.normal(peak, 5))

    elif meal_type == 2:  # Dinner
        if day_of_week in [1, 2, 4, 6]:  # Tue, Wed, Fri, Sun have stronger rush, as personally noticed by me.
            peak = 20*60  # 20:00
            if abs(t - peak) < 20 and np.random.rand() < 0.7:
                t = int(np.random.normal(peak, 10))

    return np.clip(t, start, end)

# -------------------------------
# Weather Modeling
# -------------------------------

def compute_temperature(arrival_minute):
    """
    Semi-realistic daily temperature model.
    Night ~22-24 degrees C
    Afternoon peak ~32-33 degrees C
    Based on observations from aqi.in in the first few days of March.
    """

    daily_base = np.random.normal(27, 1.0)
    amplitude = 5.5
    phase_shift = 3 * 60  # minimum around 3AM

    temp = daily_base + amplitude * np.sin(
        2 * np.pi * (arrival_minute - phase_shift) / 1440
    )

    return int(np.round(np.clip(temp, 25, 33))) # it is quite unlikely that the minimum temperature of the day (22 or so) would occur during a mealtime. 

def compute_humidity(temperature):
    """
    Mild humidity range.
    Slight inverse relation with temperature.
    """

    humidity = 60 - 0.4 * (temperature - 25)
    humidity += np.random.normal(0, 4)

    return int(np.clip(np.round(humidity), 40, 65))

def compute_wind_speed():
    """
    Wind speed in kmph.
    Moderate variability due to open campus layout.
    """

    wind = np.random.normal(10, 4)
    return round(float(np.clip(wind, 3, 25)), 1)

def compute_rain():
    """
    Rare rain events in this season.
    Binary rain indicator (0 or 1).
    """

    return np.random.choice([0, 1], p=[0.9, 0.1])

def compute_air_quality(arrival_minute):
    """
    AQI varies through day.
    Higher early morning.
    Lower afternoon.
    Range ~120-170.
    Also based on observations from aqi.in in the first few days of March.
    """

    mean_aqi = 145
    amplitude = 25
    phase_shift = 5 * 60  # peak around 5AM

    aqi = mean_aqi + amplitude * np.cos(
        2 * np.pi * (arrival_minute - phase_shift) / 1440
    )

    aqi += np.random.normal(0, 5)

    return int(np.clip(np.round(aqi), 110, 180))

# -------------------------------
# Academic Schedule Modeling
# -------------------------------

def sample_day():
    """
    Sample day index (0=Monday,...,6=Sunday)
    """
    return np.random.randint(0, 7)


def is_weekend(day):
    """
    Weekend = Saturday (5) or Sunday (6)
    """
    return 1 if day in [5, 6] else 0


def has_class(day):
    """
    Probability of having class depends on day.
    0 = class exists
    1 = no class
    """

    class_probs = {
        0: 0.9,  # Monday
        1: 0.8,  # Tuesday
        2: 0.3,  # Wednesday (low)
        3: 0.85, # Thursday
        4: 0.7,  # Friday
        5: 0.2,  # Saturday
        6: 0.0   # Sunday
    }

    has_class_today = np.random.rand() < class_probs[day]

    # assignment spec says:
    # class_day = 0 if class is there, 1 otherwise
    return 0 if has_class_today else 1

def has_deadline():
    """
    Assignment deadlines are rare.
    Binary:
    0 = no deadline
    1 = deadline
    """

    return np.random.choice([0, 1], p=[0.85, 0.15])

def sample_meal():
    """
    0 = breakfast
    1 = lunch
    2 = dinner
    """
    return np.random.choice([0, 1, 2], p=[0.33, 0.33, 0.34])


def sample_meal_type():
    """
    0 = vegetarian
    1 = non-vegetarian
    """
    return np.random.choice([0, 1], p=[0.35, 0.65])

# -------------------------------
# Sleep Behavior Modeling
# -------------------------------

def sample_sleep_wake(day):
    """
    Returns:
    rising_time (minutes from midnight)
    sleeping_time (minutes from midnight)
    """

    # Later sleep on Fri (4) and Sat (5)
    if day in [4, 5]:
        sleep = int(np.clip(np.random.normal(23*60, 30), 21*60, 1*60 + 24*60))
    else:
        sleep = int(np.clip(np.random.normal(22*60, 30), 21*60, 23*60))

    # Wake later on weekends
    if day in [5, 6]:
        rise = int(np.clip(np.random.normal(9*60, 45), 5*60, 11*60))
    else:
        rise = int(np.clip(np.random.normal(7*60 + 30, 45), 5*60, 10*60))

    return rise, sleep

# -------------------------------
# Mess Duration Model
# -------------------------------

def compute_duration(meal, weekend, class_day,
                     deadline, arrival_minute,
                     temperature, humidity,
                     rain, rise_time, sleep_time):
    """
    Compute realistic mess duration in minutes.
    """

    # Base durations
    base = {0: 12, 1: 22, 2: 28}[meal]

    duration = base

    # Weekend effect
    if weekend:
        duration += 4

    # Class effect (class_day=0 means class exists)
    if class_day == 0:
        if meal == 0:
            duration -= 5
        elif meal == 1:
            duration -= 2

    # Deadline effect
    if deadline:
        duration -= 3

    # Rush effect (stronger rush reduces duration)
    # approximate rush intensity using proximity to peak
    rush_penalty = 0
    if meal == 0:
        peaks = [8*60 + 45, 9*60 + 45]
    elif meal == 1:
        peaks = [13*60 + 15, 14*60]
    else:
        peaks = [20*60]

    for peak in peaks:
        rush_penalty += np.exp(-(arrival_minute - peak)**2 / 200)

    duration -= 4 * rush_penalty

    # Weather effect
    duration += 0.2 * (temperature - 25)
    duration -= 0.05 * (humidity - 50)
    if rain:
        duration += 2

    # Sleep effect
    if rise_time > 9*60 and meal == 0:
        duration -= 4

    if sleep_time < 22*60 and meal == 2:
        duration += 2

    # Add noise
    duration += np.random.normal(0, 3)

    duration = np.clip(duration, 5, 45)

    # Realistic reporting: whole number of minutes
    return int(np.round(duration))

# -------------------------------
# Dataset Assembly
# -------------------------------

def generate_dataset(num_samples):
    """
    Generate base dataset of num_samples rows.
    """

    student_ids, student_years = generate_students(NUM_STUDENTS)

    data_rows = []

    for _ in range(num_samples):

        # Sample student
        idx = np.random.randint(0, NUM_STUDENTS)
        student_id = student_ids[idx]
        student_year = student_years[idx]

        # Day & schedule
        day = sample_day()
        weekend_flag = is_weekend(day)
        class_flag = has_class(day)
        deadline_flag = has_deadline()

        # Meal
        meal = sample_meal()
        meal_type = sample_meal_type()

        # Arrival & environment
        arrival = sample_arrival_time(meal, day)
        temperature = compute_temperature(arrival)
        humidity = compute_humidity(temperature)
        wind = compute_wind_speed()
        rain = compute_rain()
        aqi = compute_air_quality(arrival)

        # Sleep behavior
        rise, sleep = sample_sleep_wake(day)

        # Target
        duration = compute_duration(
            meal, weekend_flag, class_flag,
            deadline_flag, arrival,
            temperature, humidity,
            rain, rise, sleep
        )

        row = {
            "student_id": student_id,
            "student_year": student_year,
            "day_of_week": day,
            "meal_time": meal,
            "meal_type": meal_type,
            "is_weekend": weekend_flag,
            "class_day": class_flag,
            "assignment_deadline": deadline_flag,
            "arrival_time_min": arrival,
            "temperature": temperature,
            "humidity": humidity,
            "wind_speed": wind,
            "rain": rain,
            "air_quality": aqi,
            "rising_time": rise,
            "sleeping_time": sleep,
            "mess_duration": duration
        }

        data_rows.append(row)
    
    print("\nMeal time distribution in generated dataset:")     # debugging
    print(pd.DataFrame(data_rows)["meal_time"].value_counts())  # debugging

    return pd.DataFrame(data_rows)

# -------------------------------
# Noisy Dataset Expansion
# -------------------------------

def generate_noisy_dataset(df_raw, target_size=5000):
    """
    Expand dataset to target_size rows and add Gaussian noise
    to selected continuous features.
    """

    repeats = int(np.ceil(target_size / len(df_raw)))

    # Duplicate dataset
    df_large = pd.concat([df_raw] * repeats, ignore_index=True)

    # Trim to exact size
    df_large = df_large.iloc[:target_size].copy()

    # Add Gaussian noise
    df_large["temperature"] += np.random.normal(0, 1, target_size)
    df_large["humidity"] += np.random.normal(0, 2, target_size)
    df_large["wind_speed"] += np.random.normal(0, 1, target_size)
    df_large["air_quality"] += np.random.normal(0, 4, target_size)
    df_large["rising_time"] += np.random.normal(0, 10, target_size)
    df_large["sleeping_time"] += np.random.normal(0, 10, target_size)

    # Reapply realistic rounding & bounds
    df_large["temperature"] = df_large["temperature"].round().clip(25, 33).astype(int)
    df_large["humidity"] = df_large["humidity"].round().clip(40, 65).astype(int)
    df_large["wind_speed"] = df_large["wind_speed"].round(1).clip(3, 25)
    df_large["air_quality"] = df_large["air_quality"].round().clip(110, 180).astype(int)
    df_large["rising_time"] = df_large["rising_time"].round().clip(300, 660).astype(int)
    df_large["sleeping_time"] = df_large["sleeping_time"].round().clip(1260, 1500).astype(int)

    return df_large

# -------------------------------
# Save Dataset
# -------------------------------

def save_dataset(df, folder_name):
    """
    Save dataset with timestamp.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, "..", "data", folder_name)

    os.makedirs(data_path, exist_ok=True)

    file_path = os.path.join(data_path, f"dataset_{folder_name}_{timestamp}.csv")

    df.to_csv(file_path, index=False)

    print(f"Saved dataset to: {file_path}")

if __name__ == "__main__":

    print("Generating raw dataset...")
    df_raw = generate_dataset(NUM_SAMPLES)

    save_dataset(df_raw, "raw")

    print("Raw dataset shape:", df_raw.shape)
    print(df_raw.head())
    print("Temp range:", df_raw["temperature"].min(), df_raw["temperature"].max())
    print("AQI range:", df_raw["air_quality"].min(), df_raw["air_quality"].max())

    print("\nGenerating noisy dataset...")
    df_noisy = generate_noisy_dataset(df_raw, 5000)

    save_dataset(df_noisy, "noisy")

    print("Noisy dataset shape:", df_noisy.shape)