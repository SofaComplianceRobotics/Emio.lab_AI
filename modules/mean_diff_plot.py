#!/usr/bin/env python3
"""
Read a CSV (default: blueleg_beam_real_sphere1018.csv), compute the Euclidean distance
between end-effector and real positions row-wise, print the mean difference and plot
both the per-row distances and their running mean.

Usage:
    python scripts/mean_diff_plot.py [path/to/file.csv]

The script will look for columns with names that commonly indicate end-effector and real
positions. If not found, pass full path and the script will try to infer columns.

Outputs:
 - prints mean distance
 - saves plot to mean_diff_plot.png in current directory
"""

import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV = Path(__file__).parents[1] / 'data' / 'blueleg_beam_real_sphere1018.csv'

# Possible column name patterns for the end-effector and real positions
EE_PATTERNS = [r'ee', r'end.?eff', r'end_effector', r'endEffector', r'Effector\sposition']
REAL_PATTERNS = [r'real', r'groundtruth', r'gt', r'meas', r'measured', r'reReal\sPositionalpos']

COORD_SUFFIXES = [r'x', r'y', r'z']


def find_columns(df):
    """Try to infer the 3D position columns for end-effector and real position.
    Returns tuples (ee_cols, real_cols) where each is [xcol,ycol,zcol] or None.
    """
    cols = df.columns.tolist()
    lower_cols = [c.lower() for c in cols]

    def match_group(patterns):
        for p in patterns:
            # try full group like ee_x, ee_y, ee_z or ee.x
            regex = re.compile(rf"{p}")
            matches = [c for c in cols if regex.search(c.lower())]
            if len(matches) >= 3:
                # try to order by suffix x,y,z
                ordered = [None, None, None]
                for c in matches:
                    s = c.lower()[-1]
                    if s in 'xyz':
                        idx = 'xyz'.index(s)
                        ordered[idx] = c
                if all(ordered):
                    return ordered
                # fallback: return first 3
                return matches[:3]
        return None

    ee_cols = match_group(EE_PATTERNS)
    real_cols = match_group(REAL_PATTERNS)

    # If patterns didn't match, fallback heuristics: look for any 3 columns that look like pos
    if ee_cols is None:
        # look for groups like pos_x pos_y pos_z
        pos_regex = re.compile(r'pos.*([xyz])$')
        matches = [c for c in cols if pos_regex.search(c.lower())]
        if len(matches) >= 3:
            ee_cols = matches[:3]

    return ee_cols, real_cols


def compute_distances(df, ee_cols, real_cols):
    ee = np.array(df[ee_cols].to_list())
    real = np.array(df[real_cols].to_list())
    diffs = ee - real
    dists = np.linalg.norm(diffs, axis=1)
    distsX = diffs[:, 0]
    distsY = diffs[:, 1]
    distsZ = diffs[:, 2]
    return dists, distsX, distsY, distsZ


def running_mean(x, N=50):
    if len(x) < N:
        N = max(1, len(x)//2)
    return np.convolve(x, np.ones(N)/N, mode='same')


def main(argv):
    if len(argv) > 1:
        path = Path(argv[1])
    else:
        path = DEFAULT_CSV

    if not path.exists():
        print(f"CSV file not found: {path}\nPlease provide the path to the CSV file as an argument.")
        sys.exit(2)


    df = pd.read_csv(path, skiprows=8, sep=";")
    numeric_const_pattern = r'[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    rx = re.compile(numeric_const_pattern, re.VERBOSE)
    # parse first column as list
    dtypes = df.dtypes.to_dict()
    if dtypes[df.columns[0]] == object:
        df[df.columns[0]] = df[df.columns[0]].apply(lambda s: [float(x) for x in rx.findall(s.strip("[]"))])
        df[df.columns[2]] = df[df.columns[2]].apply(lambda s: [float(x) for x in s.strip("[]").split(", ")])

    print(df)

    ee_cols = df.columns[0]
    real_cols = df.columns[2]

    if ee_cols is None or real_cols is None:
        print("Failed to infer end-effector or real position columns from CSV. Columns found:")
        for c in df.columns:
            print("  ", c)
        sys.exit(3)

    print("Using columns:\n  end-effector:", ee_cols, "\n  real:", real_cols)

    dists, distsX, distsY,distsZ = compute_distances(df, ee_cols, real_cols)
    mean_dist = float(np.mean(dists))
    print(f"Mean Euclidean distance between end-effector and real position: {mean_dist:.6f}")
    print(f"Mean distance components: X={np.mean(distsX):.6f}, Y={np.mean(distsY):.6f}, Z={np.mean(distsZ):.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(dists, label='per-sample distance')
    plt.plot(distsX, label='X component', alpha=0.5)
    plt.plot(distsY, label='Y component', alpha=0.5)
    plt.plot(distsZ, label='Z component', alpha=0.5)
    plt.plot(running_mean(dists, N=50), label='running mean (N=50)', linewidth=2)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.title('End-effector vs Real position distance')
    plt.legend()

    # plot in 3D the ee and real positions
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ee = np.array(df[ee_cols].to_list())
    real = np.array(df[real_cols].to_list())
    ax.scatter(ee[:, 0], ee[:, 1], ee[:, 2], label='end-effector', s=5)
    ax.scatter(real[:, 0], real[:, 1], real[:, 2], label='real', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('End-effector and Real positions')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main(sys.argv)
