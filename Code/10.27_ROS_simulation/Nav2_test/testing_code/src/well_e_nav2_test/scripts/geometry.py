"""Independent 2D footprint overlap and clearance checks for recorded physical poses."""
import math


def footprint(pose, points):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x+c*px-s*py, y+s*px+c*py) for px, py in points]


def rectangle(center, size):
    x, y = center
    hx, hy = size[0]/2, size[1]/2
    return [(x-hx, y-hy), (x+hx, y-hy), (x+hx, y+hy), (x-hx, y+hy)]


def edges(poly):
    return zip(poly, poly[1:]+poly[:1])


def overlap(a, b):
    for poly in (a, b):
        for p, q in edges(poly):
            axis = (-(q[1]-p[1]), q[0]-p[0])
            pa = [x*axis[0]+y*axis[1] for x, y in a]
            pb = [x*axis[0]+y*axis[1] for x, y in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


def point_segment(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    ratio = max(0.0, min(1.0, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.hypot(p[0]-a[0]-ratio*dx, p[1]-a[1]-ratio*dy)


def clearance(a, b):
    if overlap(a, b):
        return 0.0
    return min([point_segment(p, x, y) for p in a for x, y in edges(b)] +
               [point_segment(p, x, y) for p in b for x, y in edges(a)])


def yaw(q):
    return math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
