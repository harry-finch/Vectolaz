"""Courbes ouvertes avec angles conservés et boucles fermées non circulaires."""

import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from raster2vector import Point, GraphEdge, SkeletonGraph, BezierPath, reconstruct_geometry, svg_document
from raster2vector.bezier import fit_piecewise_bezier, evaluate_cubic


def graph(points, closed=False):
    return SkeletonGraph(300, 300, (), (GraphEdge(0, 0, 0 if closed else 1, tuple(points)),))


def assert_source_bound(path, tolerance):
    for curve in path.segments:
        source = np.array([(p.x, p.y) for p in path.source_points[curve.source_start_index:curve.source_end_index+1]])
        times = np.array(curve.source_parameters)
        for i in range(len(source)-1):
            fractions = np.linspace(0, 1, 21)
            values = evaluate_cubic(curve, times[i]+fractions*(times[i+1]-times[i]))
            line = source[i]+fractions[:, None]*(source[i+1]-source[i])
            assert np.linalg.norm(values-line, axis=1).max() <= curve.error_bound+1e-8
    assert path.error_bound <= tolerance


@pytest.mark.parametrize("reverse", [False, True])
def test_curve_and_sharp_corner_kept(reverse):
    t = np.linspace(0, 1, 121)
    points = tuple(Point(float(20+100*v), float(20+70*v*v)) for v in t)
    points += tuple(Point(float(120-60*v), 90) for v in t[1:])
    if reverse:
        points = points[::-1]
    source_graph = graph(points)
    path = reconstruct_geometry(source_graph, 1, detect_arcs=False).primitives[0]
    assert isinstance(path, BezierPath)
    assert Point(120, 90) in [curve.end for curve in path.segments]+[curve.start for curve in path.segments]
    assert path.source_points == points and not path.is_loop
    assert_source_bound(path, 1)
    assert source_graph == graph(points)


@pytest.mark.parametrize("tolerance", [1, 2])
def test_closed_ellipse_becomes_bounded_closed_bezier(tolerance):
    theta = np.linspace(0, 2*math.pi, 241)
    points = tuple(Point(float(130+80*np.cos(t)), float(130+35*np.sin(t))) for t in theta[:-1])
    points += (points[0],)
    result = reconstruct_geometry(graph(points, True), tolerance)
    path = result.primitives[0]
    assert isinstance(path, BezierPath) and path.is_loop
    assert path.segments[0].start == path.segments[-1].end == points[0]
    assert_source_bound(path, tolerance)
    document = ET.fromstring(svg_document(result))
    commands = document[0][0].get("d")
    assert " C " in commands and commands.endswith(" Z")
    assert reconstruct_geometry(graph(points, True), tolerance) == result


def test_source_indices_and_mandatory_contact():
    points = tuple(Point(float(i), float(i*i/100)) for i in range(101))
    path = fit_piecewise_bezier(points, 1, [50], source_edge_id=7)
    assert path.source_edge_id == 7
    assert any(c.source_end_index == 50 for c in path.segments)
    assert_source_bound(path, 1)
    with pytest.raises(ValueError, match="indices"):
        fit_piecewise_bezier(points, 1, [102])


def test_closed_opt_out_and_no_invented_closure():
    t = np.linspace(0, 2*math.pi, 121)[:-1]
    points = tuple(Point(float(120+60*np.cos(a)), float(120+30*np.sin(a))) for a in t)
    assert not isinstance(reconstruct_geometry(graph(points, True), 1).primitives[0], BezierPath)
    points += (points[0],)
    assert not isinstance(reconstruct_geometry(graph(points, True), 1, detect_beziers=False).primitives[0], BezierPath)
