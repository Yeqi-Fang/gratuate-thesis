import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time

class PETSimulation3D:
    def __init__(self, 
                 radius=100,         # cylinder radius
                 n_rings=8,          # how many rings along z
                 n_det_per_ring=36,  # how many detectors in each ring
                 height=200):
        """
        Sets up a 3D cylindrical PET scanner (finite height),
        but doesn't scatter-plot the entire phantom.

        Instead, we'll accumulate the emission points in one subplot as events happen.
        """
        self.radius = radius
        self.height = height
        self.z_positions = np.linspace(-height/2, height/2, n_rings)
        self.n_rings = n_rings
        self.n_det_per_ring = n_det_per_ring

        # (x, y, z) for each detector
        self.detectors = self._create_detectors_3d()

        # 3D phantom volume for random sampling only (not displayed all at once)
        self.phantom_size = 0.8 * radius
        self.phantom_3d = self._create_phantom_3d()  # Nx4 => (x,y,z,val)

        # Prepare random sampling
        self.phantom_prob = self.phantom_3d[:, 3] / np.sum(self.phantom_3d[:, 3])

        # We'll store all emission points (for the left subplot) across frames
        self.all_emit_x = []
        self.all_emit_y = []
        self.all_emit_z = []

    # --------------------------------------------------------------------------
    #               CREATE DETECTORS & PHANTOM
    # --------------------------------------------------------------------------
    def _create_detectors_3d(self):
        """Return an array of shape (N,3) with the 3D positions of all detectors."""
        detectors = []
        for zc in self.z_positions:
            for k in range(self.n_det_per_ring):
                theta = 2*np.pi * k / self.n_det_per_ring
                x = self.radius * np.cos(theta)
                y = self.radius * np.sin(theta)
                z = zc
                detectors.append([x, y, z])
        return np.array(detectors)  # shape (n_rings*n_det_per_ring, 3)

    def _create_phantom_3d(self):
        """
        Create a 3D phantom as random points in a sphere (plus hotspots).
        We'll store them as (x,y,z,val).
        """
        n_points = 20000
        coords = []
        for _ in range(n_points):
            while True:
                x = np.random.uniform(-self.phantom_size, self.phantom_size)
                y = np.random.uniform(-self.phantom_size, self.phantom_size)
                z = np.random.uniform(-self.phantom_size, self.phantom_size)
                # Must lie within a sphere
                if x*x + y*y + z*z <= self.phantom_size**2:
                    val = 1.0
                    # Example hotspot near (20,20,20)
                    if (x-20)**2 + (y-20)**2 + (z-20)**2 < 15**2:
                        val = 300.0
                    coords.append((x, y, z, val))
                    break
        return np.array(coords)  # shape (n_points,4)

    # --------------------------------------------------------------------------
    #               RANDOM EMISSION + DETECTOR INTERSECTION
    # --------------------------------------------------------------------------
    def _sample_emission_point(self):
        """Pick a random point from the 3D phantom, weighted by 'val'."""
        idx = np.random.choice(len(self.phantom_3d), p=self.phantom_prob)
        x, y, z, _ = self.phantom_3d[idx]
        return x, y, z

    def _sample_random_direction_3d(self):
        """
        Generate a random direction uniformly on the unit sphere.
        """
        u = np.random.uniform(-1, 1)  # cos(theta)
        phi = np.random.uniform(0, 2*np.pi)
        sin_theta = np.sqrt(1 - u*u)
        dx = sin_theta * np.cos(phi)
        dy = sin_theta * np.sin(phi)
        dz = u
        return dx, dy, dz

    def _intersect_line_finite_cylinder(self, x0, y0, z0, dx, dy, dz):
        """
        Intersect parametric line with FINITE cylinder x^2+y^2=R^2,
        z in [-height/2, +height/2]. Return up to two intersection points.
        """
        A = dx*dx + dy*dy
        B = 2*(x0*dx + y0*dy)
        C = x0*x0 + y0*y0 - self.radius*self.radius
        disc = B*B - 4*A*C
        if A<1e-12 or disc<0:
            return []

        t1 = (-B + np.sqrt(disc)) / (2*A)
        t2 = (-B - np.sqrt(disc)) / (2*A)
        
        zmin = -self.height/2
        zmax =  self.height/2
        pts = []
        for t in [t1, t2]:
            xI = x0 + dx*t
            yI = y0 + dy*t
            zI = z0 + dz*t
            if zmin <= zI <= zmax:
                pts.append((xI, yI, zI))
        return pts

    def _nearest_detector_index(self, x, y, z):
        """Find nearest detector among self.detectors to point (x,y,z)."""
        dist_sq = (self.detectors[:,0]-x)**2 \
                + (self.detectors[:,1]-y)**2 \
                + (self.detectors[:,2]-z)**2
        return np.argmin(dist_sq)

    def generate_event(self):
        """
        Generate one 3D emission event:
          1) pick random emission in phantom
          2) pick random direction
          3) intersect with finite cylinder
          4) if <2 hits => None
          5) else find nearest detectors => return event
        """
        x0, y0, z0 = self._sample_emission_point()
        dx, dy, dz = self._sample_random_direction_3d()

        pts = self._intersect_line_finite_cylinder(x0, y0, z0, dx, dy, dz)
        if len(pts) < 2:
            return None
        
        p1, p2 = pts[:2]
        d1 = self._nearest_detector_index(*p1)
        d2 = self._nearest_detector_index(*p2)
        return {
            'emission': (x0, y0, z0),
            'p1': p1,
            'p2': p2,
            'det1': d1,
            'det2': d2
        }

    # --------------------------------------------------------------------------
    #               ANIMATION
    # --------------------------------------------------------------------------
    def create_animation(self, n_frames=100, interval=50):
        """
        Two 3D subplots:

         LEFT: starts empty. Each new event adds a green dot at (x0, y0, z0),
               and we accumulate them over time.

         RIGHT: same as before: detectors in red, line for the event, 
                active detectors in blue, emission point in green.
        """
        fig = plt.figure(figsize=(14,6))

        # -------------------- LEFT 3D SUBPLOT (accumulating emission dots) ----
        ax1 = fig.add_subplot(1,2,1, projection='3d')
        ax1.set_title('Emission Points (accumulating)')
        ax1.set_xlim(-1.2*self.radius, 1.2*self.radius)
        ax1.set_ylim(-1.2*self.radius, 1.2*self.radius)
        ax1.set_zlim(-1.2*self.radius, 1.2*self.radius)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # We keep adding points to this scatter as events come in
        emission_history_scatter = ax1.scatter([], [], [], c='green', s=8, alpha=0.7)

        # -------------------- RIGHT 3D SUBPLOT (Detector + current event) -----
        ax2 = fig.add_subplot(1,2,2, projection='3d')
        ax2.set_title('3D PET Detector')
        ax2.set_xlim(-1.2*self.radius, 1.2*self.radius)
        ax2.set_ylim(-1.2*self.radius, 1.2*self.radius)
        ax2.set_zlim(-self.height/2 - 10, self.height/2 + 10)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Plot all detectors in red
        ax2.scatter(
            self.detectors[:,0],
            self.detectors[:,1],
            self.detectors[:,2],
            c='red', s=10, alpha=0.6, label='Detectors'
        )
        self._draw_cylinder_outline(ax2)
        ax2.legend()

        # Dynamic objects in the right subplot
        emission_scatter_right = ax2.scatter([], [], [], c='green', s=50)
        path_line, = ax2.plot([], [], [], 'y-', lw=2, alpha=0.5)
        active_det_scatter = ax2.scatter([], [], [], c='blue', s=80)

        # init() function
        def init():
            # left subplot
            emission_history_scatter._offsets3d = ([], [], [])

            # right subplot
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

        def update(frame):
            ev = self.generate_event()
            if ev is None:
                # Skip if not 2 intersections
                return (
                    emission_history_scatter,
                    emission_scatter_right,
                    path_line,
                    active_det_scatter
                )
            
            # Unpack event
            x0, y0, z0 = ev['emission']
            p1 = ev['p1']
            p2 = ev['p2']
            det1 = ev['det1']
            det2 = ev['det2']

            # 1) LEFT subplot: accumulate emission points
            self.all_emit_x.append(x0)
            self.all_emit_y.append(y0)
            self.all_emit_z.append(z0)
            emission_history_scatter._offsets3d = (
                self.all_emit_x,
                self.all_emit_y,
                self.all_emit_z
            )

            # 2) RIGHT subplot: 
            #    - emission (green)
            emission_scatter_right._offsets3d = ([x0], [y0], [z0])

            #    - path line (p1 -> emission -> p2)
            path_line.set_data(
                [p1[0], x0, p2[0]],
                [p1[1], y0, p2[1]]
            )
            path_line.set_3d_properties([p1[2], z0, p2[2]])

            #    - active detectors
            dpos1 = self.detectors[det1]
            dpos2 = self.detectors[det2]
            active_det_scatter._offsets3d = (
                [dpos1[0], dpos2[0]],
                [dpos1[1], dpos2[1]],
                [dpos1[2], dpos2[2]]
            )

            ax2.set_title(f'3D PET Detector (Frame {frame+1}/{n_frames})')
            return (
                emission_history_scatter,
                emission_scatter_right,
                path_line,
                active_det_scatter
            )

        ani = animation.FuncAnimation(
            fig, update,
            init_func=init,
            frames=n_frames,
            interval=interval,
            blit=False
        )

        # Save as MP4
        writer = animation.FFMpegWriter(fps=20, bitrate=3000)
        ani.save('pet_sim_3d_accumulate.mp4', writer=writer)

        plt.close(fig)

    def _draw_cylinder_outline(self, ax, n_lines=10):
        """Draw a wireframe of the finite cylinder boundary for reference."""
        zmin, zmax = -self.height/2, self.height/2
        angles = np.linspace(0, 2*np.pi, n_lines, endpoint=False)
        for ang in angles:
            x = self.radius*np.cos(ang)
            y = self.radius*np.sin(ang)
            ax.plot([x,x],[y,y],[zmin,zmax],'k-',alpha=0.2)

        circle_pts = 100
        circle_thetas = np.linspace(0, 2*np.pi, circle_pts)
        for zc in [zmin, zmax]:
            xc = self.radius * np.cos(circle_thetas)
            yc = self.radius * np.sin(circle_thetas)
            zc_ = np.full_like(xc, zc)
            ax.plot(xc, yc, zc_, 'k-', alpha=0.2)


# ------------------------------ MAIN ---------------------------------
if __name__ == '__main__':    
    t1 = time.perf_counter()
    np.random.seed(42)
    sim3d = PETSimulation3D(
        radius=100, 
        n_rings=8, 
        n_det_per_ring=36, 
        height=200
    )
    sim3d.create_animation(n_frames=1000, interval=50)
    t2 = time.perf_counter()
    print(f"Elapsed time: {t2-t1:.2f} sec")