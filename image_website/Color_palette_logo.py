import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import os

# Function to draw a sphere with a transparent background
def draw_sphere(ax, angle):
    # Create a mesh grid for spherical coordinates
    u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]

    # Parametric equations for a sphere
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    # Create the color map: we'll map the u angle to colors (rainbow)
    colors = plt.cm.rainbow(u / (2 * np.pi))

    # Plot the sphere with the color map
    ax.plot_surface(x, y, z, facecolors=colors, rstride=5, cstride=5, alpha=0.8, edgecolor='none')

    # Remove the axes for a clean look
    ax.set_axis_off()

    # Set the view angle for rotation
    ax.view_init(elev=20, azim=angle)

# Create a list to store frames for the GIF
frames = []

# Generate and save the frames for the rotating rainbow sphere with transparent background
for angle in range(0, 360, 5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_sphere(ax, angle)

    # Save each frame as a PNG file with transparent background
    plt.savefig(f'sphere_{angle}.png', bbox_inches='tight', pad_inches=0, transparent=True)

    # Convert the figure to a NumPy array and store it as a frame
    frames.append(imageio.imread(f'sphere_{angle}.png'))
    plt.close()

# Save frames as a GIF with a transparent background
imageio.mimsave('rotating_rainbow_sphere.gif', frames, duration=0.1, loop=0, palettesize=256)

# Clean up the generated PNG files
for file in os.listdir():
    if file.startswith('sphere_') and file.endswith('.png'):
        os.remove(file)

print("GIF created as rotating_rainbow_sphere.gif with a transparent background")
