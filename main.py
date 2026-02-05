import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
import os
import ast
from pathlib import Path
from datetime import datetime
import functions as fem
import ast

CONFIG_PATH = "config.txt"
RES_DIR = Path("Results")
RES_DIR.mkdir(parents=True, exist_ok=True)


def create_config_file(file_path: str):
    default_config_content = """# Default Configuration
# L - total length (in meters)
# H - total height (in meters)
# n_elements_x - number of elements along x-axis
# n_elements_y - number of elements along y-axis
# shape - shape of elements
# mat_bounds_x - x of materials interface location (in meters)
# mat_bounds_y - y of materials interface location (in meters)
# mat_ord - visual representation of materials' positions (ARRAY)
# materials - thermal conductivity k of each material
# left - boundary condition on left side of area (format: left = [type] [value])
# right - boundary condition on right side of area (format: right = [type] [value])
# top - boundary condition on top of area (format: [top] = type [value])
# bottom - boundary condition on bottom of area (format: bottom = [type] [value])
# y_plot - y coordinate of nodes that are considered during plotting 1-D plot (in meters)

L = 1.0
H = 1.0
n_elements_x = 4
n_elements_y = 4
shape = triangle

mat_bounds_x = 0.5
mat_bounds_y = 0.5
mat_ord = [[1, 2], [2, 1]]
materials = 1:100.0, 2:300.0

left = dirichlet 100.0
right = dirichlet 300.0
top = neumann 0.0
bottom = neumann 0.0

y_plot = None
"""

    try:
        with open(file_path, "w") as file:
            file.write(default_config_content)
            return True
    except IOError:
        return False

def read_from_file(file_path: str):
    parameters = {
        "L": 1.0, "H": 1.0,
        "n_elements_x": 10, "n_elements_y": 10,
        "shape": 'triangle',
        "mat_bounds_x": [], "mat_bounds_y": [],
        "mat_ord": [], "materials": {},
        "y_plot": None,
        "bc_config": {}
    }

    if not os.path.exists(file_path):
        print(f"\nWARNING: config file does not exist, running with default configuration\n"
              f"Exemplary file created!")
        if not create_config_file(file_path):
            raise IOError("ERROR: unable to create config file.\nTerminating...")

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line: 
            continue
    
        if "=" in line:  
            key, value = map(str.strip, line.split("=", 1))
            
            if key in ["L", "H"]:
                parameters[key] = float(value)

            elif key == "y_plot":
                if value == "None":
                    parameters[key] = None
                else:
                    parameters[key] = float(value)
            
            elif key in ["n_elements_x","n_elements_y"]:
                parameters[key] = int(value)
            
            elif key == "shape":
                parameters["shape"] = value
            
            elif key == "mat_bounds_x":
                values = value.replace(",", " ").split()
                parameters["mat_bounds_x"] = [float(x) for x in values]

            elif key == "mat_bounds_y":
                values = value.replace(",", " ").split()
                parameters["mat_bounds_y"] = [float(x) for x in values]
            
            elif key == "mat_ord":
                parameters["mat_ord"] = np.array(ast.literal_eval(value), dtype=int)
            
            elif key == "materials":
                materials = {}
                pairs = value.split(',')
                for pair in pairs:
                    if ":" in pair:
                        parts = pair.split(':')
                        m_id = int(parts[0].strip())
                        m_val = float(parts[1].strip())
                        materials[m_id] = m_val
                parameters["materials"] = materials
            
            elif key in ["right", "left", "top", "bottom"]:
                condition_type, val = value.split()
                parameters["bc_config"][key] = {
                    "type": condition_type,
                    "value": float(val)
                }

    if parameters["y_plot"] is not None:
        if not (0 <= parameters["y_plot"] <= parameters["H"]):
            parameters["y_plot"] = 0.5 * parameters['H']
            print(f"\nWARNING: y for 1-D plot exceeds the are bounds, "
                  f"setting y to y={parameters['y_plot']}")

    return parameters

def save_output_file(node_coords: np.ndarray, temperature: np.ndarray, parameters: dict):
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = RES_DIR / f"output_{stamp}.txt"

    x = node_coords[:, 0].astype(float)
    y = node_coords[:, 1].astype(float)
    T = temperature.astype(float)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"parameters:\n{parameters}\n\n")
        f.write(f"x [m], y [m],  T [K] \n")
        for i in range(T.size):
            f.write(f"({x[i]:4.2f}, {y[i]:4.2f}) = {T[i]:.3f}\n")

    return out_path

def plot_1D(nodes_coords: np.ndarray, temperature: np.ndarray, target_y: float):
    y_nodes = nodes_coords[:, 1]

    closest_idx = (np.abs(y_nodes - target_y)).argmin()
    found_y = y_nodes[closest_idx]

    indices = np.where(np.isclose(y_nodes, found_y))[0]

    xs = nodes_coords[indices, 0]
    ts = temperature[indices]

    sorted_order = np.argsort(xs)
    xs_sorted = xs[sorted_order]
    ts_sorted = ts[sorted_order]
    
    plt.figure(figsize=(8, 6))
    plt.plot(xs_sorted, ts_sorted, 'o-', label='FEM Node Values')

    plt.title(f'Temperature vs X-coordinate of Nodes for y={target_y}')
    plt.xlabel('Position $x$ [m]')
    plt.ylabel('Temperature [K]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_2D_rect(n_elements_x: int, n_elements_y: int, nodes_coords: np.ndarray, temperature: np.ndarray):
    nx_nodes = n_elements_x + 1
    ny_nodes = n_elements_y + 1

    x_nodes = nodes_coords[:, 0]
    y_nodes = nodes_coords[:, 1]

    X = x_nodes.reshape(ny_nodes, nx_nodes)
    Y = y_nodes.reshape(ny_nodes, nx_nodes)
    T = temperature.reshape(ny_nodes, nx_nodes)

    plt.figure(figsize=(10, 8))
    img = plt.pcolormesh(X, Y, T, shading='gouraud', cmap='jet')
    cbar = plt.colorbar(img, format='%.1f')
    cbar.locator = ticker.MaxNLocator(nbins=10)
    cbar.update_ticks()
    cbar.set_label('Temperature [K]')
    plt.title('2D Temperature Distribution (Rectangular Elements)')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$y$ [m]')
    plt.axis('equal')
    plt.show()

def plot_2D_triangle(elements: np.ndarray, nodes_coords: np.ndarray, temperature: np.ndarray):
    x_nodes = nodes_coords[:, 0]
    y_nodes = nodes_coords[:, 1]

    triangles = elements[:, :3].astype(int)

    plt.figure(figsize=(10, 8))
    img = plt.tripcolor(x_nodes, y_nodes, triangles, temperature,
                        shading='gouraud', cmap='jet')
    cbar = plt.colorbar(img, format='%.1f')
    cbar.locator = ticker.MaxNLocator(nbins=10)
    cbar.update_ticks()
    cbar.set_label('Temperature [K]')
    plt.title('2D Temperature Distribution (Triangular Elements)')
    plt.xlabel('$x$ [m]')
    plt.ylabel('$y$ [m]')
    plt.axis('equal')
    plt.show()


def main():
    np.set_printoptions(threshold=sys.maxsize, precision=2, suppress=True, linewidth=200)
    
    parameters = read_from_file(CONFIG_PATH)
    print(f"\nRunning with parameters:\n{parameters}")
    
    # Mesh
    nodes_coords, elements, global_node_ids = fem.generate_mesh(
        parameters["n_elements_x"], parameters["n_elements_y"], parameters["mat_ord"], 
        parameters["mat_bounds_x"], parameters["mat_bounds_y"], shape=parameters["shape"], 
        L=parameters["L"], H=parameters["H"]
    )
    ny_nodes, nx_nodes = global_node_ids.shape
    parameters["n_elements_x"] = nx_nodes - 1
    parameters["n_elements_y"] = ny_nodes - 1
    n_nodes = len(nodes_coords)

    # Global assembly
    K_global, F_global = fem.global_assembly(nodes_coords, elements, parameters["materials"], shape=parameters["shape"])

    # PRINTING BEFORE BC
    if nx_nodes < 10 and ny_nodes < 10:
        print(f"\n\nK global matrix:\n{K_global}")
        print(f"\n\nF global vector:\n{F_global}")

    # Boundary conditions application
    K_global, F_global = fem.apply_boundary_conditions(
        K_global, F_global, global_node_ids, nodes_coords, parameters["bc_config"]
    )

    # Solution
    temperature = np.linalg.solve(K_global, F_global)

    save_output_file(nodes_coords, temperature, parameters)

    if parameters["shape"] == "rect":
        plot_2D_rect(parameters["n_elements_x"], parameters["n_elements_y"], nodes_coords, temperature)
        if parameters["y_plot"] is not None:
            plot_1D(nodes_coords, temperature, parameters["y_plot"])
    elif parameters["shape"] == "triangle":
        plot_2D_triangle(elements, nodes_coords, temperature)
        if parameters["y_plot"] is not None:
            plot_1D(nodes_coords, temperature, parameters["y_plot"])


if __name__ == "__main__":
    main()