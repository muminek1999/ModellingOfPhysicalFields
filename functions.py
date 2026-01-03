import numpy as np


def generate_mesh(n_elements_x: int, n_elements_y: int, mat_ord: list[int], mat_bounds: list[float],
                  shape: str = 'triangle', L: float = 1.0, H: float = 1.0):
    nx_nodes = n_elements_x + 1
    ny_nodes = n_elements_y + 1
    n_nodes = nx_nodes * ny_nodes

    x = np.linspace(0, L, nx_nodes)
    y = np.linspace(0, H, ny_nodes)

    if n_elements_x < len(mat_bounds) + 2:
        n_elements_x += 2
        print(f"WARNING: number of elements on x-axis is too small. Number of elements is now {n_elements_x}")

    for bound in mat_bounds:
        idx = (np.abs(x - bound)).argmin()
        if 0 < idx < n_elements_x:
            x[idx] = bound
        elif idx == 0:
            x[idx + 1] = bound
        else:
            x[idx - 1] = bound

    x = np.sort(x)
    xv, yv = np.meshgrid(x, y)
    nodes_coords = np.column_stack((xv.flatten(), yv.flatten()))

    elements = []
    global_node_ids = np.arange(n_nodes).reshape(ny_nodes, nx_nodes)
    all_bounds = [0] + sorted(mat_bounds) + [L]

    if shape == 'triangle':
        for j in range(n_elements_y):
            for i in range(n_elements_x):
                x_center = (x[i] + x[i + 1]) * 0.5
                mat_idx = 0
                for k in range(len(all_bounds)):
                    if all_bounds[k] <= x_center <= all_bounds[k + 1]:
                        mat_idx = k
                        break
                mat_id = mat_ord[mat_idx]

                n1 = global_node_ids[j, i]          # Bottom-Left
                n2 = global_node_ids[j, i + 1]      # Bottom-Right
                n3 = global_node_ids[j + 1, i + 1]  # Top-Right
                n4 = global_node_ids[j + 1, i]      # Top-Left

                elements.append([n1, n2, n4, mat_id])
                elements.append([n2, n3, n4, mat_id])
        return nodes_coords, np.array(elements), global_node_ids

    elif shape == 'rect':
        for j in range(n_elements_y):
            for i in range(n_elements_x):
                x_center = (x[i] + x[i + 1]) * 0.5
                mat_idx = 0
                for k in range(len(all_bounds)):
                    if all_bounds[k] <= x_center <= all_bounds[k + 1]:
                        mat_idx = k
                        break
                mat_id = mat_ord[mat_idx]

                n1 = global_node_ids[j, i]          # Bottom-Left
                n2 = global_node_ids[j, i + 1]      # Bottom-Right
                n3 = global_node_ids[j + 1, i + 1]  # Top-Right
                n4 = global_node_ids[j + 1, i]      # Top-Left

                elements.append([n1, n2, n3, n4, mat_id])
        return nodes_coords, np.array(elements), global_node_ids

    else:
        raise ValueError('shape must be either "triangle" or "rect" (generate_mesh function)')


def element_stiffness(element: np.ndarray, nodes_coords: np.ndarray, k: float, shape: str = 'triangle'):
    if shape == 'triangle':
        x, y = get_global_coords(nodes_coords, element, shape)

        b = np.array([
            y[1] - y[2],    # b1 = y2 - y3
            y[2] - y[0],    # b2 = y3 - y1
            y[0] - y[1]     # b3 = y1 - y2
        ])

        c = np.array([
            x[2] - x[1],    # c1 = x3 - x2
            x[0] - x[2],    # c2 = x1 - x3
            x[1] - x[0]     # c3 = x2 - x1
        ])

        area = ((x[1]*y[2] - x[2]*y[1]) - (x[0]*y[2] - x[2]*y[0]) + (x[0]*y[1] - x[1]*y[0])) / 2
        k_matrix_local = np.zeros((3, 3))

        if area <= 1e-12:
            raise ValueError('element area is negative or zero')

        for i in range(3):
            for j in range(3):
                k_matrix_local[i, j] = (1 / (4 * area)) * (b[i]*b[j] + c[i]*c[j])
        return k * k_matrix_local

    elif shape == 'rect':
        x, y = get_global_coords(nodes_coords, element, shape)

        hx = np.abs(x[1] - x[0])
        hy = np.abs(y[3] - y[0])

        if hx <= 1e-12 or hy <= 1e-12:
            raise ValueError('element dimension is negative or zero')

        k11 = (hx**2 + hy**2) / (3 * hx * hy)
        k12 = (hx**2 - 2*hy**2) / (6 * hx * hy)
        k13 = -(hx**2 + hy**2) / (6 * hx * hy)
        k14 = -(2*hx**2 - hy**2) / (6 * hx * hy)

        k_matrix_local = np.array([
            [k11, k12, k13, k14],
            [k12, k11, k14, k13],
            [k13, k14, k11, k12],
            [k14, k13, k12, k11]
        ])
        return k * k_matrix_local

    else:
        raise ValueError('shape must be either "triangle" or "rect" (element_stiffness function)')


def global_assembly(nodes_coords: np.ndarray,
                    elements: np.ndarray,
                    materials: dict,
                    shape: str = 'triangle'):
    n_nodes = len(nodes_coords)
    k_matrix = np.zeros((n_nodes, n_nodes))
    f_vector = np.zeros(n_nodes)

    for element in elements:
        mat_id = int(element[-1])
        k_val = materials.get(mat_id, 1.0)

        global_indices = element[:-1]

        k_matrix_local = element_stiffness(element, nodes_coords, k_val, shape)

        for i in range(len(global_indices)):
            I = int(global_indices[i])          # Global row index
            for j in range(len(global_indices)):
                J = int(global_indices[j])      # Global column index

                k_matrix[I, J] += k_matrix_local[i, j]

    return k_matrix, f_vector


def apply_dirichlet(k_matrix: np.ndarray, f_vector: np.ndarray, node_list: np.ndarray, val: float):
    penalty_num = 1.0e15

    for node in node_list:
        k_matrix[node, node] += penalty_num
        f_vector[node] += penalty_num * val

    return k_matrix, f_vector


def apply_neumann(f_vector: np.ndarray, nodes_coords: np.ndarray, edges: np.ndarray, val: float):
    for edge in edges:
        node_i = edge[0]
        node_j = edge[1]

        x1, y1 = nodes_coords[node_i]
        x2, y2 = nodes_coords[node_j]

        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        flux_contribution = 0.5 * (val * length)

        f_vector[node_i] += flux_contribution
        f_vector[node_j] += flux_contribution

    return f_vector


def apply_boundary_conditions(k_matrix: np.ndarray, f_vector: np.ndarray, global_node_ids: np.ndarray,
                              nodes_coords: np.ndarray, bc_config: dict):
    bounds = {
        "left":     global_node_ids[:, 0],      # First column (x=0)
        "right":    global_node_ids[:, -1],     # Last column (x=L)
        "top":      global_node_ids[-1, :],     # Last row (y=H)
        "bottom":   global_node_ids[0, :]       # First row (y=0)
    }

    for side_name, condition in bc_config.items():
        node_list = bounds[side_name]
        bc_type = condition['type']
        val = condition['value']

        if bc_type == 'dirichlet':
            k_matrix, f_vector = apply_dirichlet(k_matrix, f_vector, node_list, val)

        elif bc_type == 'neumann':
            edges = []
            for i in range(len(node_list) - 1):
                n1 = node_list[i]
                n2 = node_list[i + 1]
                edges.append([n1, n2])

            f_vector = apply_neumann(f_vector, nodes_coords, np.array(edges), val)

    return k_matrix, f_vector


def get_global_coords(nodes_coords: np.ndarray, element: np.ndarray, shape: str = 'triangle'):
    if shape == 'triangle':
        n1, n2, n3, mat_id = element
        x1, y1 = nodes_coords[n1]
        x2, y2 = nodes_coords[n2]
        x3, y3 = nodes_coords[n3]

        x = [x1, x2, x3]
        y = [y1, y2, y3]
        return np.array(x), np.array(y)

    elif shape == 'rect':
        n1, n2, n3, n4, mat_id = element
        x1, y1 = nodes_coords[n1]
        x2, y2 = nodes_coords[n2]
        x3, y3 = nodes_coords[n3]
        x4, y4 = nodes_coords[n4]

        x = [x1, x2, x3, x4]
        y = [y1, y2, y3, y4]
        return np.array(x), np.array(y)

    else:
        raise ValueError('shape must be either "triangle" or "rect" (get_global_coords function)')