import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import concurrent.futures
import multiprocessing
import os

# Ensure FFmpeg is available
matplotlib.use("Agg")  # Use a non-interactive backend suitable for script execution

def simulate_event(radius, height, phantom_3d, phantom_prob, detectors):
    """
    Simulate a single PET event.

    Parameters:
    - radius: Cylinder radius.
    - height: Cylinder height.
    - phantom_3d: Array of phantom points with shape (N, 4).
    - phantom_prob: Probability distribution for selecting phantom points.
    - detectors: Array of detector positions with shape (M, 3).

    Returns:
    - Dict with event details or None if invalid.
    """
    # 1. Random emission point
    idx = np.random.choice(len(phantom_3d), p=phantom_prob)
    x0, y0, z0, _ = phantom_3d[idx]

    # 2. Random direction uniformly on the unit sphere
    u = np.random.uniform(-1, 1)  # cos(theta)
    phi = np.random.uniform(0, 2 * np.pi)
    sin_theta = np.sqrt(1 - u * u)
    dx = sin_theta * np.cos(phi)
    dy = sin_theta * np.sin(phi)
    dz = u

    # 3. Intersect with finite cylinder
    A = dx * dx + dy * dy
    B = 2 * (x0 * dx + y0 * dy)
    C = x0 * x0 + y0 * y0 - radius * radius
    disc = B * B - 4 * A * C

    if A < 1e-12 or disc < 0:
        return None  # No valid intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-B + sqrt_disc) / (2 * A)
    t2 = (-B - sqrt_disc) / (2 * A)

    zmin = -height / 2
    zmax = height / 2
    pts = []
    for t in [t1, t2]:
        xI = x0 + dx * t
        yI = y0 + dy * t
        zI = z0 + dz * t
        if zmin <= zI <= zmax:
            pts.append((xI, yI, zI))

    if len(pts) < 2:
        return None  # Less than two valid intersections

    p1, p2 = pts[:2]

    # 4. Find nearest detectors for p1 and p2 separately
    # Compute distances for p1
    dist1_sq = np.sum((detectors - p1) ** 2, axis=1)
    det1 = np.argmin(dist1_sq)

    # Compute distances for p2
    dist2_sq = np.sum((detectors - p2) ** 2, axis=1)
    det2 = np.argmin(dist2_sq)

    return {
        'emission': (x0, y0, z0),
        'p1': p1,
        'p2': p2,
        'det1': det1,
        'det2': det2
    }

class PETSimulation3D:
    def __init__(self, 
                 radius=100,         # Cylinder radius
                 n_rings=8,          # Number of rings along z-axis
                 n_det_per_ring=36,  # Number of detectors per ring
                 height=200,         # Total height of the cylinder
                 n_points=20000):    # Number of phantom points
        """
        Initialize the 3D PET Simulation.

        Parameters:
        - radius: Cylinder radius.
        - n_rings: Number of detector rings along the z-axis.
        - n_det_per_ring: Number of detectors per ring.
        - height: Total height of the cylinder.
        - n_points: Number of points in the phantom.
        """
        self.radius = radius
        self.height = height
        self.n_rings = n_rings
        self.n_det_per_ring = n_det_per_ring
        self.n_points = n_points

        # Detector geometry
        self.detectors = self._create_detectors_3d()

        # Phantom generation
        self.phantom_3d = self._create_phantom_3d()
        self.phantom_prob = self.phantom_3d[:, 3] / np.sum(self.phantom_3d[:, 3])

        # Storage for events
        self.all_events = []
        self.all_emit_x = []
        self.all_emit_y = []
        self.all_emit_z = []

    def _create_detectors_3d(self):
        """Create detector positions in a cylindrical arrangement."""
        detectors = []
        z_positions = np.linspace(-self.height / 2, self.height / 2, self.n_rings)
        for zc in z_positions:
            for k in range(self.n_det_per_ring):
                theta = 2 * np.pi * k / self.n_det_per_ring
                x = self.radius * np.cos(theta)
                y = self.radius * np.sin(theta)
                z = zc
                detectors.append([x, y, z])
        return np.array(detectors)  # Shape: (n_rings * n_det_per_ring, 3)

    def _create_phantom_3d(self):
        """Generate a 3D phantom with hotspots."""
        coords = []
        for _ in range(self.n_points):
            while True:
                x = np.random.uniform(-0.8 * self.radius, 0.8 * self.radius)
                y = np.random.uniform(-0.8 * self.radius, 0.8 * self.radius)
                z = np.random.uniform(-0.8 * self.radius, 0.8 * self.radius)
                if x**2 + y**2 + z**2 <= (0.8 * self.radius)**2:
                    val = 1.0
                    # Add hotspots
                    if (x - 20)**2 + (y - 20)**2 + (z - 20)**2 < 15**2:
                        val = 300.0
                    elif (x + 30)**2 + (y + 30)**2 + (z + 30)**2 < 20**2:
                        val = 200.0
                    elif (x - 10)**2 / 400 + (y + 20)**2 / 100 <= 1:
                        val = 400.0
                    coords.append((x, y, z, val))
                    break
        return np.array(coords)  # Shape: (n_points, 4)

    def generate_all_events_parallel(self, n_frames):
        """
        Generate all events in parallel and write valid detector pairs to a text file.

        Parameters:
        - n_frames: Number of events to generate.
        """
        print("Starting parallel event generation...")
        tasks = [
            (
                self.radius, 
                self.height, 
                self.phantom_3d, 
                self.phantom_prob, 
                self.detectors
            )
            for _ in range(n_frames)
        ]

        # Determine the number of workers based on CPU cores
        num_workers = multiprocessing.cpu_count()
        print(f"Using {num_workers} parallel workers.")

        # Ensure the output file is empty before writing
        with open("coincidences.txt", "w") as f:
            f.write("")  # Clear the file

        # Use ProcessPoolExecutor for CPU-bound tasks
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(simulate_event, *task) for task in tasks]

            # Collect results as they complete
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                ev = future.result()
                self.all_events.append(ev)
                if ev is not None:
                    # Write to the text file
                    with open("coincidences.txt", "a") as f:
                        f.write(f"{ev['det1']} {ev['det2']}\n")
                if i % 10 == 0 or i == n_frames:
                    print(f"Processed {i}/{n_frames} events.")

        print("Event generation completed.")

    def create_animation(self, n_frames=100, interval=50):
        """
        Create and save the PET simulation animation with two 3D subplots.

        Parameters:
        - n_frames: Number of frames/events in the animation.
        - interval: Delay between frames in milliseconds.
        """
        print("Starting animation creation...")

        fig = plt.figure(figsize=(16, 8))

        # -------------------- LEFT 3D SUBPLOT (Accumulated Emissions) --------------------
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_title('Accumulated Emissions')
        ax1.set_xlim(-1.2 * self.radius, 1.2 * self.radius)
        ax1.set_ylim(-1.2 * self.radius, 1.2 * self.radius)
        ax1.set_zlim(-1.2 * self.radius, 1.2 * self.radius)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')

        emission_history_scatter = ax1.scatter([], [], [], c='green', s=20, alpha=0.6)

        # -------------------- RIGHT 3D SUBPLOT (Detector & Current Event) --------------------
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.set_title('3D PET Detector')
        ax2.set_xlim(-1.2 * self.radius, 1.2 * self.radius)
        ax2.set_ylim(-1.2 * self.radius, 1.2 * self.radius)
        ax2.set_zlim(-self.height / 2 - 10, self.height / 2 + 10)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_zlabel('Z (mm)')

        # Plot all detectors in red
        ax2.scatter(
            self.detectors[:, 0],
            self.detectors[:, 1],
            self.detectors[:, 2],
            c='red', s=20, alpha=0.6, label='Detectors'
        )
        self._draw_cylinder_outline(ax2)
        ax2.legend()

        # Dynamic elements in the right subplot
        emission_scatter_right = ax2.scatter([], [], [], c='green', s=50)
        path_line, = ax2.plot([], [], [], 'y-', lw=2, alpha=0.5)
        active_det_scatter = ax2.scatter([], [], [], c='blue', s=80)

        # Initialize animation
        def init():
            emission_history_scatter._offsets3d = ([], [], [])
            emission_scatter_right._offsets3d = ([], [], [])
            path_line.set_data([], [])
            path_line.set_3d_properties([])
            active_det_scatter._offsets3d = ([], [], [])
            return (
                emission_history_scatter,
                emission_scatter_right,
                path_line,
                active_det_scatter
            )

        # Update function for each frame
        def update(frame):
            if frame >= len(self.all_events):
                return (
                    emission_history_scatter,
                    emission_scatter_right,
                    path_line,
                    active_det_scatter
                )

            ev = self.all_events[frame]
            if ev is None:
                # Invalid event; skip updating
                return (
                    emission_history_scatter,
                    emission_scatter_right,
                    path_line,
                    active_det_scatter
                )

            # Unpack event details
            x0, y0, z0 = ev['emission']
            p1 = ev['p1']
            p2 = ev['p2']
            det1 = ev['det1']
            det2 = ev['det2']

            # 1. Update the left subplot (Accumulated Emissions)
            self.all_emit_x.append(x0)
            self.all_emit_y.append(y0)
            self.all_emit_z.append(z0)
            emission_history_scatter._offsets3d = (
                self.all_emit_x,
                self.all_emit_y,
                self.all_emit_z
            )

            # 2. Update the right subplot (Detector & Current Event)
            # a. Current emission point
            emission_scatter_right._offsets3d = ([x0], [y0], [z0])

            # b. Path line: p1 -> emission -> p2
            path_line.set_data(
                [p1[0], x0, p2[0]],
                [p1[1], y0, p2[1]]
            )
            path_line.set_3d_properties(
                [p1[2], z0, p2[2]]
            )

            # c. Active detectors
            dpos1 = self.detectors[det1]
            dpos2 = self.detectors[det2]
            active_det_scatter._offsets3d = (
                [dpos1[0], dpos2[0]],
                [dpos1[1], dpos2[1]],
                [dpos1[2], dpos2[2]]
            )

            ax2.set_title(f'3D PET Detector (Frame {frame + 1}/{n_frames})')
            return (
                emission_history_scatter,
                emission_scatter_right,
                path_line,
                active_det_scatter
            )

        # Create the animation
        ani = animation.FuncAnimation(
            fig, update,
            init_func=init,
            frames=n_frames,
            interval=interval,
            blit=False
        )

        # Save the animation as MP4
        print("Saving animation to 'pet_sim_3d_accumulate_parallel.mp4'...")
        writer = animation.FFMpegWriter(fps=20, bitrate=3000)
        ani.save('pet_sim_3d_accumulate_parallel.mp4', writer=writer)
        print("Animation saved successfully.")

        plt.close(fig)

    def _draw_cylinder_outline(self, ax, n_lines=20):
        """Draw a wireframe of the finite cylinder boundary for reference."""
        zmin, zmax = -self.height / 2, self.height / 2
        angles = np.linspace(0, 2 * np.pi, n_lines, endpoint=False)
        for ang in angles:
            x = self.radius * np.cos(ang)
            y = self.radius * np.sin(ang)
            ax.plot([x, x], [y, y], [zmin, zmax], 'k-', alpha=0.2)

        # Top and bottom circles
        circle_pts = 100
        circle_thetas = np.linspace(0, 2 * np.pi, circle_pts)
        for zc in [zmin, zmax]:
            xc = self.radius * np.cos(circle_thetas)
            yc = self.radius * np.sin(circle_thetas)
            zc_ = np.full_like(xc, zc)
            ax.plot(xc, yc, zc_, 'k-', alpha=0.2)

# ------------------------------ MAIN ---------------------------------
if __name__ == '__main__':
    # Protect the multiprocessing entry point
    # This is essential on Windows to prevent recursive spawning of subprocesses
    # when using ProcessPoolExecutor
    # Reference: https://docs.python.org/3/library/multiprocessing.html#safe-importing-the-main-module

    # Ensure that the text file is empty before starting
    if os.path.exists("coincidences.txt"):
        os.remove("coincidences.txt")

    np.random.seed(42)
    sim3d = PETSimulation3D(
        radius=100, 
        n_rings=8, 
        n_det_per_ring=36, 
        height=200,
        n_points=20000
    )

    # Define the number of frames/events
    n_frames = 10000
    interval = 50  # in milliseconds

    # 1. Generate all events in parallel
    sim3d.generate_all_events_parallel(n_frames)

    # 2. Create and save the animation
    sim3d.create_animation(n_frames=n_frames, interval=interval)
