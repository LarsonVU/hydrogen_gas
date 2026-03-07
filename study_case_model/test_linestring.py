import networkx as nx
from shapely.geometry import MultiLineString, LineString, Point

A = Point((1,3))
B = Point ((0,0))

line_1 = LineString([A, B])
print(line_1)

