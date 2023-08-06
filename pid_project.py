import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Update the t_end and n_steps values for 10 iterations
t_start = 0
t_end = 500
n_steps = 50
dt = (t_end - t_start) / n_steps
time_values = [t_start + i*dt for i in range(n_steps)]

# We will use a constant target value (10)
setpoint_values = [10 for _ in time_values]

# Our starting value is 0, it will be updated in the next steps with the values coming from the PID control
measurements = [0]
errors = [setpoint_values[0] - measurements[0]]

# Set the settings of the PID controller (according to the values used in the example table)
Kp = 0.5
Ki = 0.1
Kd = 0.2

integral_values = [0]
proportional_values = [0]
derivative_values = [0]
pid_values = [0]
integral = 0
previous_error = 0  # Variable to store the previous error value
calculation_steps = []

for i in range(1, n_steps):
    e = errors[-1]
    integral += e*dt
    proportional = Kp*e
    derivative = Kd*(e - previous_error)/dt
    integral_contribution = Ki*integral
    pid = proportional + integral_contribution + derivative

    # Add the PID output to our current value
    measurements.append(measurements[-1] + pid)
    
    # Calculate the error for the next step
    error = setpoint_values[i] - measurements[-1]
    errors.append(error)
    
    integral_values.append(integral_contribution)
    proportional_values.append(proportional)
    derivative_values.append(derivative)
    pid_values.append(pid)
    
    calculation_steps.append([f'{proportional:.2f} (0.5*{e:.2f})', f'{integral_contribution:.2f} (0.1*{integral:.2f})', f'{derivative:.2f} (0.2*({e:.2f}-{previous_error:.2f}))'])

    previous_error = e

# Function to apply 5 point moving average filter
def moving_average_filter(data, window_size):
    filtered_data = []
    for i in range(len(data)):
        if i < window_size:
            # For the first points, we take the average of all available points
            filtered_data.append(np.mean(data[:i+1]))
        else:
            # For later points, we take the average of previous points with a specific window size
            filtered_data.append(np.mean(data[i-window_size+1:i+1]))
    return filtered_data

sigma = 2  # Parameter determining the width of the Gaussian filter. You can change the level of filtering by adjusting this value.

errors_filtered = gaussian_filter1d(errors, sigma)
proportional_values_filtered = gaussian_filter1d(proportional_values, sigma)
integral_values_filtered = gaussian_filter1d(integral_values, sigma)
derivative_values_filtered = gaussian_filter1d(derivative_values, sigma)
pid_values_filtered = gaussian_filter1d(pid_values, sigma)
measurements_filtered = gaussian_filter1d(measurements, sigma)


# Draw the change of filtered error, proportional value, integral value, derivative value and PID output over time
plt.figure(figsize=(10,6))
plt.plot(time_values, errors_filtered, label='Error (filtered)')
plt.plot(time_values, proportional_values_filtered, label='Proportional (filtered)')
plt.plot(time_values, integral_values_filtered, label='Integral (filtered)')
plt.plot(time_values, derivative_values_filtered, label='Derivative (filtered)')
plt.plot(time_values, pid_values_filtered, label='PID output (filtered)')
plt.plot(time_values, measurements_filtered, label='Measurements')

# Limit x-axis to the specified values
plt.xlim(0, 100)  # You can change these values according to your needs

plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Change in Filtered Error, Proportional, Integral, Derivative, and PID Output Over Time')
plt.grid(True)
plt.show()


data = {
    'Iteration': list(range(n_steps)),
    'Target': setpoint_values,
    'C.V.': measurements,
    'Error': errors,
    'P': proportional_values,
    'I': integral_values,
    'D': derivative_values,
    'Total Correction': pid_values
}
df = pd.DataFrame(data)

# Let's split the calculation steps into parts: Proportional, Integral and Derivative
calculation_steps_np = np.array(calculation_steps)
proportional_steps = calculation_steps_np[:, 0]
integral_steps = calculation_steps_np[:, 1]
derivative_steps = calculation_steps_np[:, 2]

# Update the DataFrame
df['P'] = [''] + list(proportional_steps)
df['I'] = [''] + list(integral_steps)
df['D'] = [''] + list(derivative_steps)

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
    pd.set_option('display.float_format', '{:.2f}'.format)
    print("C.V. = Current Value, T.C. = Total Correction")  # Explanation line
    print(df)