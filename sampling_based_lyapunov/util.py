import numpy as np
import casadi as ca


def format_matrix(matrix, environment="pmatrix", formatter=str):
    """Format a matrix using LaTeX syntax"""

    if not isinstance(matrix, np.ndarray):
        try:
            matrix = np.array(matrix)
        except Exception:
            raise TypeError("Could not convert to Numpy array")

    if len(shape := matrix.shape) == 1:
        matrix = matrix.reshape(1, shape[0])
    elif len(shape) == 2:
        for i in range(shape[0]):
            for j in range(shape[1]):
                matrix[i, j] = np.round(matrix[i, j], 4)
                if np.abs(matrix[i, j]) < 1e-5:
                    matrix[i, j] = 0
    elif len(shape) > 2:
        raise ValueError("Array must be 2 dimensional")

    body_lines = [" & ".join(map(formatter, row)) for row in matrix]

    body = "\\\\\n".join(body_lines)
    return f"""\\begin{{{environment}}}
{body}
\\end{{{environment}}}"""


def quaternion_multiply(quaternion0, quaternion1):
    p0, p1, p2, p3 = quaternion0
    q0, q1, q2, q3 = quaternion1
    if type(quaternion0) is np.ndarray:
        return np.array([p0*q0 - p1*q1 - p2*q2 - p3*q3, p0*q1 + p1*q0 + p2*q3 - p3*q2, p0*q2 - p1*q3 + p2*q0 + p3*q1,
                         p0*q3 + p1*q2 - p2*q1 + p3*q0])
    elif type(quaternion0[0]) is ca.MX:
        return ca.vertcat(p0*q0 - p1*q1 - p2*q2 - p3*q3, p0*q1 + p1*q0 + p2*q3 - p3*q2, p0*q2 - p1*q3 + p2*q2 + p3*q1,
                          p0*q3 + p1*q2 - p2*q2 + p3*q0)
    else:
        return ValueError
