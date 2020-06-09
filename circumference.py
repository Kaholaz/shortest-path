import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from numba import njit
import argparse
import cProfile
import pstats
from typing import Tuple, List, Any

default_n = 100000


def main(n=default_n) -> None:
    """
    The program works by first sorting all the points based on
    their angle to the centroid of the points. When the points
    are sorted, it determines the angle between all the points
    in this sequence. It then eliminates all the points from
    the sequence that makes the line through all the points
    "bend" outwards from the centre of the set of points.
    """
    points: List[Tuple[Any, ...]] = get_points(n)
    centroid: Tuple[float, float] = (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )
    # Sorting based on the points angle to the centroid of the points.
    # This orders the points in the the correct order for
    # the points along the circumference.
    circ_points: List[Tuple[Any, ...]] = sorted(
        points, key=(lambda p: get_rotation(p, centroid))
    )
    p: List[Tuple[Any, ...]] = circ_points  # Shorthand for circ_points

    start = 0
    i: int = 0  # index of a point along the circle
    d: int = +1  # d is direction: +1 for forward, -1 for backwards
    halt: bool = False

    p_length: int = len(circ_points)
    pr: cProfile.Profile = cProfile.Profile()
    pr.enable()
    iterations: int = 0
    roundtrips: int = 0

    while True:
        iterations += 1
        p_0: Tuple[Any, ...] = p[i]
        p_1: Tuple[Any, ...] = p[(i + d) % p_length]
        p_2: Tuple[Any, ...] = p[(i + 2 * d) % p_length]
        # vector between p_0 and p_1
        v1: Tuple[float, float] = vector_between_points(p_0, p_1)
        # vector between p_1 and p_2
        v2: Tuple[float, float] = vector_between_points(p_1, p_2)
        # if the vectors "bend" outwards, the middle point is
        # deleted. The direction needs to be accounted for.
        if (get_angle_between_two_vecotors(v1, v2) > math.pi) == (d == 1):
            halt = False
            j: int = (i + d) % p_length
            del circ_points[j]
            p_length -= 1
            # You need to account for a shift in indexes due to deleting
            # one of the elements in circ_points.
            if j < i:
                i -= 1
        else:
            i = (i + d) % p_length
            if i == start:  # this means the program has made a roundtrip
                roundtrips += 1
                # halts if the programmed havn't made changes to
                # circ_points during the roundtrip
                if halt:
                    break
                # changes directions and resets variables for another roundtrip
                else:
                    halt = True
                    d *= -1

    ps = pstats.Stats(pr).sort_stats("time")
    ps.print_stats(10)
    print(f"iterations: {iterations}")
    print(f"roundtrips: {roundtrips}")
    mpl.style.use("seaborn")
    circ_points.append(circ_points[0])
    plt.scatter(*np.array(points).T)
    plt.plot(*np.array(circ_points).T)
    plt.show()


@njit
def get_rotation(point: Tuple[float, float], centroid: Tuple[float, float]) -> float:
    return math.atan2(point[1] - centroid[1], point[0] - centroid[0])


@njit
def get_angle_between_two_vecotors(
    v1: Tuple[float, float], v2: Tuple[float, float]
) -> float:
    angle = math.atan2(v2[1], v2[0]) - math.atan2(v1[1], v1[0])
    return angle % (2 * math.pi)


@njit
def vector_between_points(point1, point2):
    return (point2[0] - point1[0], point2[1] - point1[1])


def get_points(n) -> List[Tuple[Any, ...]]:
    return [tuple(p) for p in np.random.normal(0, 10, (n, 2))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--number_of_points",
        help="Provides the number of random points to calculate",
        type=int,
        default=default_n,
    )
    args = parser.parse_args()
    main(args.number_of_points)
