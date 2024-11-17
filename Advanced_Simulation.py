# Python Program for Space Rocket Trajectory and Course Correction

# Libraries Needed:
# NumPy: For numerical operations and vector calculations.
# SciPy: For solving differential equations.
# Matplotlib: For visualization (optional, if you want to plot the rocket's path).

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_earth = 5.972e24  # Mass of Earth (kg)
R_earth = 6371000  # Radius of Earth (m)
target_distance = 10000000  # Target distance from Earth's center (m)

# Rocket class to store properties
class Rocket:
    def __init__(self, x, y, velocity, angle, mass, thrust):
        self.x = x  # Position on x-axis (m)
        self.y = y  # Position on y-axis (m)
        self.velocity = velocity  # Speed (m/s)
        self.angle = angle  # Direction of motion (radians)
        self.mass = mass  # Mass of the rocket (kg)
        self.thrust = thrust  # Thrust force (N)

    def get_position(self):
        return np.array([self.x, self.y])

# Gravitational force between two bodies
def gravity_force(rocket_pos):
    r = np.linalg.norm(rocket_pos)  # Distance from the Earth's center
    force = G * M_earth * rocket_pos / r**3  # Gravitational force vector (N)
    return force

# Rocket motion differential equations
def rocket_dynamics(t, state, rocket):
    x, y, vx, vy = state  # Unpack position and velocity
    rocket_pos = np.array([x, y])
    
    # Gravitational force acting on the rocket
    gravity = gravity_force(rocket_pos)

    # Thrust force in the direction of the rocket's angle (2D plane)
    thrust_x = rocket.thrust * np.cos(rocket.angle)
    thrust_y = rocket.thrust * np.sin(rocket.angle)

    # Net force (Thrust - Gravitational force)
    net_force = np.array([thrust_x, thrust_y]) - gravity

    # Rocket dynamics: acceleration = force / mass (F = ma)
    ax = net_force[0] / rocket.mass
    ay = net_force[1] / rocket.mass

    # Return derivatives (velocity and acceleration)
    return [vx, vy, ax, ay]

# PID Controller for course correction
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Simulation setup
rocket = Rocket(x=R_earth + 100000, y=0, velocity=0, angle=np.pi/4, mass=5000, thrust=10000)  # 100km above Earth
target_position = np.array([target_distance, 0])  # Target position (on the x-axis)
pid_controller = PIDController(kp=0.5, ki=0.05, kd=0.1)

# Initial conditions (x, y, vx, vy)
initial_state = [rocket.x, rocket.y, rocket.velocity * np.cos(rocket.angle), rocket.velocity * np.sin(rocket.angle)]

# Time settings
t_span = np.linspace(0, 10000, 10000)  # 10,000 seconds, adjust for accuracy
dt = t_span[1] - t_span[0]

# Solve the rocket's dynamics using SciPy's ODE solver
def run_simulation():
    def dynamics(t, state):
        error = np.linalg.norm(np.array([state[0], state[1]]) - target_position)  # Distance to target
        control = pid_controller.compute(error, dt)  # Course correction
        
        # Update rocket's angle based on control input from PID
        rocket.angle += control * dt

        return rocket_dynamics(t, state, rocket)

    # Solve ODEs
    solution = integrate.solve_ivp(dynamics, (t_span[0], t_span[-1]), initial_state, t_eval=t_span)
    
    return solution

# Run the simulation
solution = run_simulation()

# Extract the results
x_vals = solution.y[0]
y_vals = solution.y[1]

# Plot the rocket's trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Rocket Trajectory")
plt.scatter(target_position[0], target_position[1], color='red', label='Target')
plt.xlabel("Distance (m)")
plt.ylabel("Height (m)")
plt.legend()
plt.title("Rocket Trajectory with Course Correction")
plt.grid(True)
plt.show()

