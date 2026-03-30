"""
NAME: Computational Geometry (Convex Hull, Line Intersection, Point-in-Polygon, Closest Pair)
TAGS: geometry, convex-hull, line-intersection, closest-pair, graham-scan
DESCRIPTION: Core computational geometry primitives for competitive programming. Includes
    Graham scan convex hull, segment/line intersection, point-in-polygon test, and the
    O(N log N) closest pair of points algorithm. Use these as a tested foundation rather
    than implementing from scratch under contest pressure.
COMPLEXITY: Convex Hull O(N log N), Closest Pair O(N log N), Point-in-Polygon O(N); Space O(N)
"""

from math import inf, sqrt, atan2
from functools import cmp_to_key

EPS = 1e-9


# ── Point / Vector Primitives ─────────────────────────────────────────────────
def cross(O, A, B):
    """Cross product of vectors OA and OB. Positive = left turn, negative = right turn."""
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

def dot(A, B, C, D):
    """Dot product of vectors AB and CD."""
    return (B[0]-A[0])*(D[0]-C[0]) + (B[1]-A[1])*(D[1]-C[1])

def dist2(A, B):
    """Squared Euclidean distance between A and B."""
    return (A[0]-B[0])**2 + (A[1]-B[1])**2

def dist(A, B):
    """Euclidean distance between A and B."""
    return sqrt(dist2(A, B))

def orientation(p, q, r):
    """
    Returns:  0 if collinear, 1 if clockwise, -1 if counterclockwise.
    Based on the sign of cross product of (q-p) and (r-p).
    """
    v = cross(p, q, r)
    if abs(v) < EPS: return 0
    return 1 if v < 0 else -1


# ── Convex Hull (Graham Scan) ─────────────────────────────────────────────────
def convex_hull(points):
    """
    Returns vertices of convex hull in counterclockwise order.
    Handles duplicates and collinear points (excluded from hull boundary by default).
    Input: list of (x, y) tuples. Requires >= 3 distinct non-collinear points.
    """
    pts = sorted(set(points))
    n = len(pts)
    if n <= 1: return pts
    if n == 2: return pts

    # Build lower hull
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Remove last point of each half because it's repeated
    return lower[:-1] + upper[:-1]

def convex_hull_area(hull):
    """Compute area of convex hull using the shoelace formula."""
    n = len(hull)
    area = 0
    for i in range(n):
        area += hull[i][0] * hull[(i+1) % n][1]
        area -= hull[(i+1) % n][0] * hull[i][1]
    return abs(area) / 2

# Example:
# pts = [(0,0),(1,1),(2,2),(0,2),(2,0)]
# hull = convex_hull(pts) -> [(0,0),(2,0),(2,2),(0,2)]  (CCW order)
# convex_hull_area(hull)  -> 4.0


# ── Segment Intersection ──────────────────────────────────────────────────────
def on_segment(p, q, r):
    """Check if r lies on segment pq (assumes collinear)."""
    return (min(p[0],q[0]) <= r[0] <= max(p[0],q[0]) and
            min(p[1],q[1]) <= r[1] <= max(p[1],q[1]))

def segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    Returns True if they share any point (including endpoints).
    """
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and on_segment(p3, p4, p1): return True
    if d2 == 0 and on_segment(p3, p4, p2): return True
    if d3 == 0 and on_segment(p1, p2, p3): return True
    if d4 == 0 and on_segment(p1, p2, p4): return True
    return False

def line_intersection(p1, p2, p3, p4):
    """
    Find intersection point of lines p1-p2 and p3-p4.
    Returns (x, y) or None if lines are parallel.
    """
    dx1, dy1 = p2[0]-p1[0], p2[1]-p1[1]
    dx2, dy2 = p4[0]-p3[0], p4[1]-p3[1]
    denom = dx1 * dy2 - dy1 * dx2
    if abs(denom) < EPS:
        return None  # Parallel lines
    dx3, dy3 = p3[0]-p1[0], p3[1]-p1[1]
    t = (dx3 * dy2 - dy3 * dx2) / denom
    return (p1[0] + t * dx1, p1[1] + t * dy1)

# Example:
# segments_intersect((0,0),(2,2),(0,2),(2,0)) -> True
# line_intersection((0,0),(2,2),(0,2),(2,0))  -> (1.0, 1.0)


# ── Point in Polygon (Ray Casting) ────────────────────────────────────────────
def point_in_polygon(point, polygon):
    """
    Check if point is inside polygon using ray casting.
    Returns: 'inside', 'boundary', or 'outside'.
    Polygon is a list of vertices in order (CW or CCW).
    """
    x, y = point
    n = len(polygon)
    inside = False

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[(i+1) % n]

        # Check if point is on the boundary
        if abs(cross((xi, yi), (xj, yj), point)) < EPS and \
           on_segment((xi, yi), (xj, yj), point):
            return 'boundary'

        # Ray casting
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside

    return 'inside' if inside else 'outside'

# Example:
# poly = [(0,0),(4,0),(4,4),(0,4)]
# point_in_polygon((2,2), poly)   -> 'inside'
# point_in_polygon((0,0), poly)   -> 'boundary'
# point_in_polygon((5,5), poly)   -> 'outside'


# ── Closest Pair of Points ────────────────────────────────────────────────────
def closest_pair(points):
    """
    Find closest pair of points in O(N log N) using divide and conquer.
    Returns (min_distance, point_A, point_B).
    """
    def closest_strip(strip, d):
        strip.sort(key=lambda p: p[1])
        min_d = d
        best = None
        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and (strip[j][1] - strip[i][1]) < min_d:
                dd = dist(strip[i], strip[j])
                if dd < min_d:
                    min_d = dd
                    best = (strip[i], strip[j])
                j += 1
        return min_d, best

    def rec(pts):
        n = len(pts)
        if n <= 3:
            best_d = inf
            best_pair = None
            for i in range(n):
                for j in range(i+1, n):
                    d = dist(pts[i], pts[j])
                    if d < best_d:
                        best_d = d
                        best_pair = (pts[i], pts[j])
            return best_d, best_pair

        mid = n // 2
        mid_x = pts[mid][0]
        dl, pl = rec(pts[:mid])
        dr, pr = rec(pts[mid:])

        if dl < dr:
            d, pair = dl, pl
        else:
            d, pair = dr, pr

        strip = [p for p in pts if abs(p[0] - mid_x) < d]
        ds, ps = closest_strip(strip, d)
        if ds < d:
            return ds, ps
        return d, pair

    if len(points) < 2:
        return inf, None
    pts = sorted(points)
    return rec(pts)

# Example:
# pts = [(0,0),(3,4),(1,1),(5,2)]
# closest_pair(pts) -> (sqrt(2), (0,0), (1,1))


# ── Polygon Area (Shoelace) ───────────────────────────────────────────────────
def polygon_area(polygon):
    """Signed area of polygon (positive if CCW). Use abs() for actual area."""
    n = len(polygon)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2

# Example:
# polygon_area([(0,0),(4,0),(4,4),(0,4)]) -> 16.0  (CCW: positive)


# ── Pick's Theorem ────────────────────────────────────────────────────────────
def picks_theorem(interior, boundary):
    """
    Area = I + B/2 - 1  where I = interior lattice pts, B = boundary lattice pts.
    Equivalently: I = Area - B/2 + 1.
    """
    return interior + boundary // 2 - 1  # returns Area given I and B

def boundary_lattice_points(polygon):
    """Count lattice points on boundary of integer-coordinate polygon."""
    from math import gcd
    n = len(polygon)
    b = 0
    for i in range(n):
        dx = abs(polygon[(i+1)%n][0] - polygon[i][0])
        dy = abs(polygon[(i+1)%n][1] - polygon[i][1])
        b += gcd(dx, dy)
    return b

# Example:
# poly = [(0,0),(4,0),(0,3)]
# B = boundary_lattice_points(poly) -> 4+3+gcd(4,3)=4+3+1=8 -> wrong, it's gcd each edge
# Area = abs(polygon_area(poly)) -> 6.0
# Interior I = Area - B/2 + 1 = 6 - 8/2 + 1 = 3
