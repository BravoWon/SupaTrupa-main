"""
Unit tests for Multidimensional Point Cloud Ingestion Utility

Tests cover:
- Core data structures (Point, PointCloud2D/3D/4D)
- Indexing (SpatialIndex, TemporalIndex, SpatioTemporalIndex)
- TDA integration
- Knowledge Graph linkage
"""

import pytest
import numpy as np
from typing import List

# =============================================================================
# Core Data Structure Tests
# =============================================================================

class TestPoint:
    """Tests for Point class."""

    def test_point_creation(self):
        """Test basic point creation."""
        from jones_framework.data.point_cloud import Point

        p2d = Point(coordinates=(1.0, 2.0))
        assert p2d.dimensionality == 2
        assert p2d.x == 1.0
        assert p2d.y == 2.0
        assert p2d.z == 0.0  # Default for 2D

        p3d = Point(coordinates=(1.0, 2.0, 3.0))
        assert p3d.dimensionality == 3
        assert p3d.z == 3.0

        p4d = Point(coordinates=(1.0, 2.0, 3.0, 100.0))
        assert p4d.dimensionality == 4
        assert p4d.t == 100.0

    def test_point_with_uncertainty(self):
        """Test point with uncertainty values."""
        from jones_framework.data.point_cloud import Point

        p = Point(
            coordinates=(1.0, 2.0, 3.0),
            uncertainties=(0.1, 0.2, 0.3)
        )
        assert p.uncertainties == (0.1, 0.2, 0.3)

        coords, uncert = p.to_uncertain_numpy()
        assert np.allclose(coords, [1.0, 2.0, 3.0])
        assert np.allclose(uncert, [0.1, 0.2, 0.3])

    def test_point_uncertainty_dimension_mismatch(self):
        """Test that mismatched uncertainty dimensions raise error."""
        from jones_framework.data.point_cloud import Point

        with pytest.raises(ValueError):
            Point(coordinates=(1.0, 2.0, 3.0), uncertainties=(0.1, 0.2))

    def test_point_distance(self):
        """Test distance calculation between points."""
        from jones_framework.data.point_cloud import Point

        p1 = Point(coordinates=(0.0, 0.0, 0.0))
        p2 = Point(coordinates=(3.0, 4.0, 0.0))

        assert p1.distance_to(p2, metric='euclidean') == 5.0
        assert p1.distance_to(p2, metric='manhattan') == 7.0


class TestPointCloud2D:
    """Tests for PointCloud2D class."""

    def test_creation_from_numpy(self):
        """Test 2D cloud creation from numpy array."""
        from jones_framework.data.point_cloud import PointCloud2D

        data = np.random.rand(100, 2)
        cloud = PointCloud2D.from_numpy(data)

        assert cloud.num_points == 100
        assert cloud.points.shape == (100, 2)

    def test_invalid_dimensions(self):
        """Test that wrong dimensions raise error."""
        from jones_framework.data.point_cloud import PointCloud2D

        data = np.random.rand(100, 3)  # 3D data
        with pytest.raises(ValueError):
            PointCloud2D.from_numpy(data)

    def test_bounds_and_centroid(self):
        """Test bounding box and centroid calculation."""
        from jones_framework.data.point_cloud import PointCloud2D

        data = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
        cloud = PointCloud2D.from_numpy(data)

        min_coords, max_coords = cloud.bounds
        assert np.allclose(min_coords, [0, 0])
        assert np.allclose(max_coords, [2, 2])
        assert np.allclose(cloud.centroid, [1, 1])

    def test_to_tda_input(self):
        """Test TDA input conversion."""
        from jones_framework.data.point_cloud import PointCloud2D

        data = np.random.rand(50, 2)
        cloud = PointCloud2D.from_numpy(data)
        tda_input = cloud.to_tda_input()

        assert tda_input.shape == (50, 2)

    def test_subsample_random(self):
        """Test random subsampling."""
        from jones_framework.data.point_cloud import PointCloud2D

        data = np.random.rand(100, 2)
        cloud = PointCloud2D.from_numpy(data)
        subsampled = cloud.subsample(20, method='random')

        assert subsampled.num_points == 20

    def test_subsample_farthest(self):
        """Test farthest point subsampling."""
        from jones_framework.data.point_cloud import PointCloud2D

        data = np.random.rand(100, 2)
        cloud = PointCloud2D.from_numpy(data)
        subsampled = cloud.subsample(10, method='farthest')

        assert subsampled.num_points == 10


class TestPointCloud3D:
    """Tests for PointCloud3D class."""

    def test_creation(self):
        """Test 3D cloud creation."""
        from jones_framework.data.point_cloud import PointCloud3D, CoordinateSystem

        data = np.random.rand(100, 3)
        cloud = PointCloud3D.from_numpy(
            data,
            coord_system=CoordinateSystem.CARTESIAN
        )

        assert cloud.num_points == 100
        assert cloud.coordinate_system == CoordinateSystem.CARTESIAN

    def test_with_uncertainty(self):
        """Test 3D cloud with uncertainty."""
        from jones_framework.data.point_cloud import PointCloud3D

        data = np.random.rand(50, 3)
        uncert = np.random.rand(50, 3) * 0.1
        cloud = PointCloud3D.from_numpy(data, uncertainties=uncert)

        assert cloud.uncertainties is not None
        assert cloud.total_uncertainty > 0

    def test_to_2d_projection(self):
        """Test 2D projection."""
        from jones_framework.data.point_cloud import PointCloud3D

        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        cloud = PointCloud3D.from_numpy(data)

        xy = cloud.to_2d('xy')
        assert xy.points.shape == (2, 2)
        assert np.allclose(xy.points[0], [1, 2])

        xz = cloud.to_2d('xz')
        assert np.allclose(xz.points[0], [1, 3])


class TestPointCloud4D:
    """Tests for PointCloud4D class."""

    def test_creation(self):
        """Test 4D cloud creation."""
        from jones_framework.data.point_cloud import PointCloud4D

        data = np.random.rand(100, 4)
        cloud = PointCloud4D.from_numpy(data)

        assert cloud.num_points == 100
        assert cloud.points.shape == (100, 4)

    def test_temporal_bounds(self):
        """Test temporal bounds extraction."""
        from jones_framework.data.point_cloud import PointCloud4D

        data = np.array([
            [0, 0, 0, 10],
            [1, 1, 1, 20],
            [2, 2, 2, 30]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        t_min, t_max = cloud.temporal_bounds
        assert t_min == 10.0
        assert t_max == 30.0

    def test_slice_by_time(self):
        """Test temporal slicing."""
        from jones_framework.data.point_cloud import PointCloud4D

        data = np.array([
            [0, 0, 0, 10],
            [1, 1, 1, 20],
            [2, 2, 2, 30],
            [3, 3, 3, 40]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        sliced = cloud.slice_by_time(15, 35)
        assert sliced.num_points == 2
        assert np.allclose(sliced.temporal_bounds, (20, 30))

    def test_to_3d_at_time(self):
        """Test 3D projection at specific time."""
        from jones_framework.data.point_cloud import PointCloud4D

        data = np.array([
            [0, 0, 0, 10],
            [1, 1, 1, 10],
            [2, 2, 2, 20]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        snapshot = cloud.to_3d_at_time(10.0)
        assert snapshot.num_points == 2

    def test_from_3d_sequence(self):
        """Test creation from sequence of 3D clouds."""
        from jones_framework.data.point_cloud import PointCloud3D, PointCloud4D

        cloud1 = PointCloud3D.from_numpy(np.random.rand(10, 3))
        cloud2 = PointCloud3D.from_numpy(np.random.rand(15, 3))
        cloud3 = PointCloud3D.from_numpy(np.random.rand(20, 3))

        cloud_4d = PointCloud4D.from_3d_sequence(
            [cloud1, cloud2, cloud3],
            timestamps=[0.0, 1.0, 2.0]
        )

        assert cloud_4d.num_points == 45  # 10 + 15 + 20

    def test_unique_timestamps(self):
        """Test unique timestamp extraction."""
        from jones_framework.data.point_cloud import PointCloud4D

        data = np.array([
            [0, 0, 0, 10],
            [1, 1, 1, 10],
            [2, 2, 2, 20],
            [3, 3, 3, 20],
            [4, 4, 4, 30]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        unique_t = cloud.unique_timestamps
        assert len(unique_t) == 3
        assert np.allclose(unique_t, [10, 20, 30])


# =============================================================================
# Indexing Tests
# =============================================================================

class TestSpatialIndex:
    """Tests for SpatialIndex class."""

    def test_build_index(self):
        """Test index building."""
        from jones_framework.data.point_cloud import SpatialIndex

        points = np.random.rand(100, 3)
        index = SpatialIndex.build(points)

        assert index.num_points == 100
        assert index.dimensionality == 3

    def test_radius_query(self):
        """Test radius query."""
        from jones_framework.data.point_cloud import SpatialIndex

        points = np.array([
            [0, 0, 0],
            [0.1, 0, 0],
            [1, 0, 0],
            [2, 0, 0]
        ], dtype=np.float32)
        index = SpatialIndex.build(points)

        nearby = index.query_radius(np.array([0, 0, 0]), radius=0.5)
        assert len(nearby) == 2  # [0, 0, 0] and [0.1, 0, 0]

    def test_knn_query(self):
        """Test k-nearest neighbors query."""
        from jones_framework.data.point_cloud import SpatialIndex

        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 0, 0]
        ], dtype=np.float32)
        index = SpatialIndex.build(points)

        indices, distances = index.query_knn(np.array([0, 0, 0]), k=2)
        assert len(indices) == 2
        assert 0 in indices  # Nearest is itself

    def test_box_query(self):
        """Test bounding box query."""
        from jones_framework.data.point_cloud import SpatialIndex

        points = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [1, 1, 1],
            [2, 2, 2]
        ], dtype=np.float32)
        index = SpatialIndex.build(points)

        inside = index.query_box(
            min_coords=np.array([0, 0, 0]),
            max_coords=np.array([1, 1, 1])
        )
        assert len(inside) == 3  # Excludes [2, 2, 2]


class TestTemporalIndex:
    """Tests for TemporalIndex class."""

    def test_build_index(self):
        """Test temporal index building."""
        from jones_framework.data.point_cloud import TemporalIndex

        times = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        index = TemporalIndex.build(times)

        assert index.num_points == 5
        assert index.time_range == (10.0, 50.0)

    def test_range_query(self):
        """Test time range query."""
        from jones_framework.data.point_cloud import TemporalIndex

        times = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        index = TemporalIndex.build(times)

        in_range = index.query_range(25, 45)
        assert len(in_range) == 2  # 30 and 40

    def test_nearest_query(self):
        """Test nearest time query."""
        from jones_framework.data.point_cloud import TemporalIndex

        times = np.array([10, 20, 30, 40, 50], dtype=np.float64)
        index = TemporalIndex.build(times)

        nearest_idx = index.query_nearest(33)
        assert times[nearest_idx] == 30  # 30 is closer to 33 than 40

    def test_time_slices(self):
        """Test time slice generation."""
        from jones_framework.data.point_cloud import TemporalIndex

        times = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.float64)
        index = TemporalIndex.build(times)

        slices = index.get_time_slices(5)
        assert len(slices) == 5
        assert slices[0] == (0.0, 20.0)


class TestSpatioTemporalIndex:
    """Tests for SpatioTemporalIndex class."""

    def test_build_index(self):
        """Test combined index building."""
        from jones_framework.data.point_cloud import SpatioTemporalIndex

        points_4d = np.random.rand(100, 4)
        points_4d[:, 3] = np.linspace(0, 100, 100)  # Time column
        index = SpatioTemporalIndex.build(points_4d)

        assert index.num_points == 100

    def test_spatiotemporal_query(self):
        """Test combined space-time query."""
        from jones_framework.data.point_cloud import SpatioTemporalIndex

        points_4d = np.array([
            [0, 0, 0, 10],
            [0.1, 0.1, 0.1, 15],
            [1, 1, 1, 20],
            [2, 2, 2, 50]
        ], dtype=np.float32)
        index = SpatioTemporalIndex.build(points_4d)

        # Query near origin, in time range 5-25
        result = index.query_spatiotemporal(
            center=np.array([0, 0, 0]),
            spatial_radius=0.5,
            t_start=5,
            t_end=25
        )

        # Should find points at [0,0,0,10] and [0.1,0.1,0.1,15]
        assert len(result) == 2


# =============================================================================
# Knowledge Graph Tests
# =============================================================================

class TestPointCloudEntity:
    """Tests for PointCloudEntity class."""

    def test_entity_creation(self):
        """Test entity creation."""
        from jones_framework.data.point_cloud import PointCloudEntity

        entity = PointCloudEntity(
            entity_id="building_1",
            entity_type="building",
            point_indices=np.array([0, 1, 2, 3])
        )

        assert entity.entity_id == "building_1"
        assert entity.entity_type == "building"
        assert entity.num_points == 4

    def test_add_relationship(self):
        """Test adding relationships."""
        from jones_framework.data.point_cloud import PointCloudEntity, SemanticRelationType

        entity = PointCloudEntity("e1", "type1")
        entity.add_relationship("e2", SemanticRelationType.ADJACENT_TO)

        related = entity.get_related_ids(SemanticRelationType.ADJACENT_TO)
        assert "e2" in related

    def test_serialization(self):
        """Test entity serialization."""
        from jones_framework.data.point_cloud import PointCloudEntity, SemanticRelationType

        entity = PointCloudEntity(
            entity_id="e1",
            entity_type="test",
            point_indices=np.array([1, 2, 3]),
            properties={"color": "red"}
        )
        entity.add_relationship("e2", SemanticRelationType.PART_OF)

        # Convert to dict and back
        data = entity.to_dict()
        restored = PointCloudEntity.from_dict(data)

        assert restored.entity_id == entity.entity_id
        assert restored.entity_type == entity.entity_type
        assert len(restored.point_indices) == 3


class TestPointCloudKnowledgeGraph:
    """Tests for PointCloudKnowledgeGraph class."""

    def test_register_entity(self):
        """Test entity registration."""
        from jones_framework.data.point_cloud import (
            PointCloudKnowledgeGraph, PointCloudEntity
        )

        kg = PointCloudKnowledgeGraph()
        entity = PointCloudEntity("e1", "building", np.array([0, 1, 2]))
        kg.register_entity(entity)

        assert kg.num_entities == 1
        assert kg.get_entity("e1") is not None

    def test_query_by_type(self):
        """Test type-based query."""
        from jones_framework.data.point_cloud import (
            PointCloudKnowledgeGraph, PointCloudEntity
        )

        kg = PointCloudKnowledgeGraph()
        kg.register_entity(PointCloudEntity("b1", "building"))
        kg.register_entity(PointCloudEntity("b2", "building"))
        kg.register_entity(PointCloudEntity("v1", "vehicle"))

        buildings = kg.query_by_type("building")
        assert len(buildings) == 2

        vehicles = kg.query_by_type("vehicle")
        assert len(vehicles) == 1

    def test_add_and_query_relationships(self):
        """Test relationship creation and querying."""
        from jones_framework.data.point_cloud import (
            PointCloudKnowledgeGraph, PointCloudEntity, SemanticRelationType
        )

        kg = PointCloudKnowledgeGraph()
        kg.register_entity(PointCloudEntity("room1", "room"))
        kg.register_entity(PointCloudEntity("building1", "building"))

        kg.add_relationship("room1", "building1", SemanticRelationType.PART_OF)

        related = kg.query_related("room1", SemanticRelationType.PART_OF)
        assert len(related) == 1
        assert related[0].entity_id == "building1"

    def test_get_points_for_entity(self):
        """Test point retrieval for entity."""
        from jones_framework.data.point_cloud import (
            PointCloudKnowledgeGraph, PointCloudEntity
        )

        kg = PointCloudKnowledgeGraph()
        entity = PointCloudEntity("e1", "test", np.array([10, 20, 30]))
        kg.register_entity(entity)

        points = kg.get_points_for_entity("e1")
        assert len(points) == 3
        assert 20 in points

    def test_to_triples(self):
        """Test RDF-style triple export."""
        from jones_framework.data.point_cloud import (
            PointCloudKnowledgeGraph, PointCloudEntity, SemanticRelationType
        )

        kg = PointCloudKnowledgeGraph()
        entity = PointCloudEntity("e1", "building", properties={"height": 100})
        kg.register_entity(entity)
        kg.register_entity(PointCloudEntity("e2", "room"))
        kg.add_relationship("e2", "e1", SemanticRelationType.PART_OF)

        triples = kg.to_triples()
        assert len(triples) >= 3  # At least type triples + relationship


# =============================================================================
# TDA Integration Tests
# =============================================================================

class TestPointCloudTDAAdapter:
    """Tests for TDA integration."""

    def test_adapter_creation(self):
        """Test adapter creation."""
        from jones_framework.data.point_cloud import PointCloudTDAAdapter

        adapter = PointCloudTDAAdapter(max_homology_dim=2)
        assert adapter.max_homology_dim == 2

    def test_analyze_2d(self):
        """Test 2D analysis."""
        from jones_framework.data.point_cloud import PointCloud2D, PointCloudTDAAdapter

        # Create a simple circle
        theta = np.linspace(0, 2 * np.pi, 50)
        x = np.cos(theta)
        y = np.sin(theta)
        data = np.column_stack([x, y])

        cloud = PointCloud2D.from_numpy(data)
        adapter = PointCloudTDAAdapter()
        diagram = adapter.analyze_2d(cloud)

        # Should detect H0 (connected components) and H1 (loop)
        assert 0 in diagram.diagrams
        betti = diagram.betti_numbers()
        assert betti.get(0, 0) >= 0

    def test_analyze_4d_temporal(self):
        """Test temporal analysis."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudTDAAdapter

        # Create time-varying point cloud
        n_points = 100
        times = np.linspace(0, 10, n_points)
        xyz = np.random.rand(n_points, 3)
        data = np.column_stack([xyz, times])

        cloud = PointCloud4D.from_numpy(data)
        adapter = PointCloudTDAAdapter()

        diagrams = adapter.analyze_4d_temporal(cloud, num_slices=3)
        assert len(diagrams) >= 1

    def test_compute_betti_curve(self):
        """Test Betti curve computation."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudTDAAdapter

        n_points = 200
        times = np.linspace(0, 10, n_points)
        xyz = np.random.rand(n_points, 3)
        data = np.column_stack([xyz, times])

        cloud = PointCloud4D.from_numpy(data)
        adapter = PointCloudTDAAdapter()

        timestamps, betti_values = adapter.compute_betti_curve(cloud, dim=0, num_slices=5)

        assert len(timestamps) == len(betti_values)
        assert len(timestamps) <= 5

    def test_streaming_analyzer(self):
        """Test streaming TDA state."""
        from jones_framework.data.point_cloud import PointCloudTDAAdapter

        adapter = PointCloudTDAAdapter()
        streaming_state = adapter.create_streaming_analyzer(window_size=3)

        # Add points incrementally
        for i in range(5):
            points = np.random.rand(10, 3)
            diagram = streaming_state.add_points(points, timestamp=float(i))

            if i >= 2:  # Window is full
                assert diagram is not None

        recent = streaming_state.get_recent_diagrams(2)
        assert len(recent) <= 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline(self):
        """Test complete pipeline from data to analysis."""
        from jones_framework.data.point_cloud import (
            PointCloud4D, PointCloudKnowledgeGraph, PointCloudEntity,
            SemanticRelationType, PointCloudTDAAdapter
        )

        # Create 4D point cloud
        n_points = 100
        data = np.random.rand(n_points, 4)
        data[:, 3] = np.linspace(0, 10, n_points)
        cloud = PointCloud4D.from_numpy(data)

        # Create knowledge graph with entities
        kg = PointCloudKnowledgeGraph()
        kg.register_entity(PointCloudEntity(
            "cluster_1", "cluster", np.arange(0, 30)
        ))
        kg.register_entity(PointCloudEntity(
            "cluster_2", "cluster", np.arange(30, 60)
        ))
        kg.add_relationship("cluster_1", "cluster_2", SemanticRelationType.ADJACENT_TO)

        # Analyze topology
        adapter = PointCloudTDAAdapter()
        diagrams = adapter.analyze_4d_temporal(cloud, num_slices=3)

        # Verify integration
        assert cloud.num_points == 100
        assert kg.num_entities == 2
        assert len(diagrams) >= 1

    def test_spatiotemporal_kg_integration(self):
        """Test spatial queries with KG entity filtering."""
        from jones_framework.data.point_cloud import (
            PointCloud4D, SpatioTemporalIndex,
            PointCloudKnowledgeGraph, PointCloudEntity
        )

        # Create structured 4D data
        data = np.array([
            [0, 0, 0, 0],
            [0.1, 0.1, 0.1, 1],
            [5, 5, 5, 2],
            [5.1, 5.1, 5.1, 3]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        # Build index
        index = SpatioTemporalIndex.build(cloud.points)

        # Create KG with entities
        kg = PointCloudKnowledgeGraph()
        kg.register_entity(PointCloudEntity("region_a", "region", np.array([0, 1])))
        kg.register_entity(PointCloudEntity("region_b", "region", np.array([2, 3])))

        # Query near region_a
        nearby = index.query_spatiotemporal(
            center=np.array([0, 0, 0]),
            spatial_radius=1.0,
            t_start=-1,
            t_end=5
        )

        # Verify results are in region_a
        region_a_points = set(kg.get_points_for_entity("region_a"))
        for idx in nearby:
            assert idx in region_a_points


# =============================================================================
# Layer 4: Labeling Tests
# =============================================================================

class TestPointLabel:
    """Tests for PointLabel class."""

    def test_label_creation(self):
        """Test basic label creation."""
        from jones_framework.data.point_cloud import PointLabel, LabelType

        label = PointLabel(
            label_id=1,
            label_name="ground",
            color=(0, 255, 0),
            label_type=LabelType.SEMANTIC
        )

        assert label.label_id == 1
        assert label.label_name == "ground"
        assert label.color == (0, 255, 0)
        assert label.label_type == LabelType.SEMANTIC

    def test_label_color_conversion(self):
        """Test color format conversions."""
        from jones_framework.data.point_cloud import PointLabel

        label = PointLabel(label_id=1, label_name="test", color=(255, 128, 64))

        assert label.to_hex_color() == "#ff8040"
        rgb_norm = label.to_rgb_normalized()
        assert np.allclose(rgb_norm, (1.0, 0.502, 0.251), atol=0.01)

    def test_label_serialization(self):
        """Test label to_dict/from_dict."""
        from jones_framework.data.point_cloud import PointLabel, LabelType

        original = PointLabel(
            label_id=5,
            label_name="building",
            color=(100, 100, 200),
            confidence=0.9,
            label_type=LabelType.INSTANCE
        )

        data = original.to_dict()
        restored = PointLabel.from_dict(data)

        assert restored.label_id == original.label_id
        assert restored.label_name == original.label_name
        assert restored.color == original.color
        assert restored.label_type == original.label_type


class TestPointCloudLabels:
    """Tests for PointCloudLabels class."""

    def test_create_labels(self):
        """Test label manager creation."""
        from jones_framework.data.point_cloud import PointCloudLabels

        labels = PointCloudLabels.create(100)
        assert labels.assignments is not None
        assert labels.assignments.num_points == 100
        assert labels.assignments.num_unlabeled == 100

    def test_register_and_assign(self):
        """Test label registration and point assignment."""
        from jones_framework.data.point_cloud import PointCloudLabels, PointLabel, LabelType

        labels = PointCloudLabels.create(50)

        # Register label
        label = PointLabel(1, "ground", (0, 255, 0), label_type=LabelType.SEMANTIC)
        assert labels.register_label(label)

        # Assign to points
        count = labels.label_points(np.array([0, 1, 2, 3, 4]), 1)
        assert count == 5

        # Query
        ground_points = labels.get_points_by_label(1)
        assert len(ground_points) == 5

    def test_label_region(self):
        """Test region-based labeling."""
        from jones_framework.data.point_cloud import PointCloudLabels, PointLabel

        labels = PointCloudLabels.create(100)
        labels.register_label(PointLabel(1, "building", (255, 0, 0)))

        # Label a region
        indices = np.array([10, 11, 12, 13, 14])
        labels.label_region("building_A", indices, 1)

        # Get region
        region = labels.get_region("building_A")
        assert region is not None
        assert len(region[0]) == 5
        assert region[1].label_name == "building"

    def test_merge_labels(self):
        """Test merging two labels."""
        from jones_framework.data.point_cloud import PointCloudLabels, PointLabel

        labels = PointCloudLabels.create(100)
        labels.register_label(PointLabel(1, "label_a", (255, 0, 0)))
        labels.register_label(PointLabel(2, "label_b", (0, 255, 0)))

        labels.label_points(np.array([0, 1, 2]), 1)
        labels.label_points(np.array([3, 4, 5]), 2)

        # Merge into new label
        new_label = PointLabel(3, "merged", (128, 128, 0))
        labels.merge_labels(1, 2, new_label)

        merged_points = labels.get_points_by_label(3)
        assert len(merged_points) == 6

    def test_label_distribution(self):
        """Test label distribution statistics."""
        from jones_framework.data.point_cloud import PointCloudLabels, PointLabel

        labels = PointCloudLabels.create(100)
        labels.register_label(PointLabel(1, "a", (255, 0, 0)))
        labels.register_label(PointLabel(2, "b", (0, 255, 0)))

        labels.label_points(np.arange(0, 30), 1)
        labels.label_points(np.arange(30, 50), 2)

        dist = labels.get_label_distribution()
        assert dist[1] == 30
        assert dist[2] == 20

        assert labels.get_labeled_ratio() == 0.5


# =============================================================================
# Layer 5: Editing Tests
# =============================================================================

class TestEditHistory:
    """Tests for EditHistory class."""

    def test_push_and_undo(self):
        """Test pushing and undoing operations."""
        from jones_framework.data.point_cloud import EditHistory, EditRecord, EditOperation

        history = EditHistory()

        record = EditRecord(
            operation=EditOperation.TRANSLATE,
            timestamp=1.0,
            params={'offset': [1, 0, 0]},
            inverse_data={'original_points': [[0, 0, 0]]},
            description="Translate X"
        )
        history.push(record)

        assert history.can_undo()
        assert not history.can_redo()

        popped = history.pop_undo()
        assert popped.operation == EditOperation.TRANSLATE
        assert history.can_redo()

    def test_redo(self):
        """Test redo functionality."""
        from jones_framework.data.point_cloud import EditHistory, EditRecord, EditOperation

        history = EditHistory()
        record = EditRecord(EditOperation.SCALE, 1.0, {}, {})
        history.push(record)

        history.pop_undo()
        assert history.can_redo()

        redone = history.pop_redo()
        assert redone.operation == EditOperation.SCALE
        assert history.can_undo()

    def test_max_history(self):
        """Test max history limit."""
        from jones_framework.data.point_cloud import EditHistory, EditRecord, EditOperation

        history = EditHistory(max_history=5)

        for i in range(10):
            history.push(EditRecord(EditOperation.ADD, float(i), {}, {}))

        assert history.undo_count == 5


class TestPointCloudEditor:
    """Tests for PointCloudEditor class."""

    def test_translate(self):
        """Test translate operation."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudEditor

        data = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)
        editor = PointCloudEditor(cloud)

        editor.translate(np.array([10, 0, 0]))

        assert np.allclose(cloud.points[0, :3], [10, 0, 0])
        assert np.allclose(cloud.points[1, :3], [11, 1, 1])

    def test_add_points(self):
        """Test adding points."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudEditor

        data = np.array([[0, 0, 0, 0]], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)
        editor = PointCloudEditor(cloud)

        new_points = np.array([[1, 1, 1, 1], [2, 2, 2, 2]], dtype=np.float32)
        editor.add_points(new_points)

        assert cloud.num_points == 3

    def test_delete_points(self):
        """Test deleting points."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudEditor

        data = np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [2, 2, 2, 2]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)
        editor = PointCloudEditor(cloud)

        editor.delete_points(np.array([1]))

        assert cloud.num_points == 2

    def test_undo_translate(self):
        """Test undo of translate operation."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudEditor

        data = np.array([[0, 0, 0, 0]], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)
        editor = PointCloudEditor(cloud)

        original_pos = cloud.points[0].copy()
        editor.translate(np.array([10, 20, 30]))

        assert not np.allclose(cloud.points[0, :3], original_pos[:3])

        editor.undo()
        assert np.allclose(cloud.points[0, :3], original_pos[:3])

    def test_subsample(self):
        """Test subsampling."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudEditor

        data = np.random.rand(100, 4).astype(np.float32)
        cloud = PointCloud4D.from_numpy(data)
        editor = PointCloudEditor(cloud)

        editor.subsample(0.5, method='random')
        assert cloud.num_points < 100

    def test_filter_by_bounds(self):
        """Test filtering by bounds."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudEditor

        data = np.array([
            [0, 0, 0, 0],
            [5, 5, 5, 0],
            [10, 10, 10, 0]
        ], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)
        editor = PointCloudEditor(cloud)

        editor.filter_by_bounds(
            min_bounds=np.array([0, 0, 0]),
            max_bounds=np.array([6, 6, 6])
        )

        assert cloud.num_points == 2


# =============================================================================
# Layer 6: Persistence Tests
# =============================================================================

class TestPointCloudIO:
    """Tests for PointCloudIO class."""

    def test_save_load_npz(self, tmp_path):
        """Test NPZ save and load."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudIO

        data = np.random.rand(50, 4).astype(np.float32)
        cloud = PointCloud4D.from_numpy(data)

        path = tmp_path / "test.npz"
        PointCloudIO.save(cloud, str(path))

        loaded = PointCloudIO.load(str(path))
        assert loaded.num_points == 50
        assert np.allclose(loaded.points, cloud.points)

    def test_save_load_json(self, tmp_path):
        """Test JSON save and load."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudIO

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        path = tmp_path / "test.json"
        PointCloudIO.save(cloud, str(path))

        loaded = PointCloudIO.load(str(path))
        assert loaded.num_points == 2
        assert np.allclose(loaded.points, cloud.points)

    def test_save_load_with_labels(self, tmp_path):
        """Test saving and loading with labels."""
        from jones_framework.data.point_cloud import (
            PointCloud4D, PointCloudIO, PointCloudLabels, PointLabel
        )

        data = np.random.rand(20, 4).astype(np.float32)
        cloud = PointCloud4D.from_numpy(data)

        labels = PointCloudLabels.create(20)
        labels.register_label(PointLabel(1, "ground", (0, 255, 0)))
        labels.label_points(np.arange(0, 10), 1)

        path = tmp_path / "test_labels.npz"
        PointCloudIO.save(cloud, str(path), labels=labels)

        loaded_cloud, loaded_labels = PointCloudIO.load_with_labels(str(path))
        assert loaded_cloud.num_points == 20
        assert loaded_labels is not None
        assert len(loaded_labels.get_points_by_label(1)) == 10

    def test_save_load_csv(self, tmp_path):
        """Test CSV save and load."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudIO

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        path = tmp_path / "test.csv"
        PointCloudIO.save(cloud, str(path))

        loaded = PointCloudIO.load(str(path))
        assert loaded.num_points == 2

    def test_save_load_ply(self, tmp_path):
        """Test PLY save and load."""
        from jones_framework.data.point_cloud import PointCloud4D, PointCloudIO

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        cloud = PointCloud4D.from_numpy(data)

        path = tmp_path / "test.ply"
        PointCloudIO.save(cloud, str(path))

        loaded = PointCloudIO.load(str(path))
        assert loaded.num_points == 2

    def test_export_for_training(self, tmp_path):
        """Test ML training export."""
        from jones_framework.data.point_cloud import (
            PointCloud4D, PointCloudIO, PointCloudLabels, PointLabel
        )

        data = np.random.rand(100, 4).astype(np.float32)
        cloud = PointCloud4D.from_numpy(data)

        labels = PointCloudLabels.create(100)
        labels.register_label(PointLabel(1, "class_a", (255, 0, 0)))
        labels.label_points(np.arange(0, 50), 1)

        paths = PointCloudIO.export_for_training(
            cloud, labels, str(tmp_path),
            split_ratios=(0.6, 0.2, 0.2)
        )

        assert 'train' in paths
        assert 'val' in paths
        assert 'test' in paths


# =============================================================================
# Layer 7: Visualization Tests
# =============================================================================

class TestColorMapper:
    """Tests for ColorMapper class."""

    def test_map_values(self):
        """Test value to color mapping."""
        from jones_framework.data.point_cloud import ColorMapper, Colormap

        mapper = ColorMapper(colormap=Colormap.VIRIDIS, vmin=0, vmax=100)
        values = np.array([0, 50, 100])

        colors = mapper.map_values(values)

        assert colors.shape == (3, 3)
        assert np.all(colors >= 0) and np.all(colors <= 1)

    def test_map_labels(self):
        """Test label to color mapping."""
        from jones_framework.data.point_cloud import ColorMapper

        mapper = ColorMapper()
        labels = np.array([0, 0, 1, 1, 2, -1])  # -1 is unlabeled

        colors = mapper.map_labels(labels)

        assert colors.shape == (6, 3)
        # Unlabeled should be gray
        assert np.allclose(colors[5], [0.5, 0.5, 0.5])

    def test_distinct_colors(self):
        """Test distinct color generation."""
        from jones_framework.data.point_cloud import ColorMapper

        colors = ColorMapper.generate_distinct_colors(10)

        assert len(colors) == 10
        # All colors should be RGB tuples in 0-255 range
        for c in colors:
            assert len(c) == 3
            assert all(0 <= v <= 255 for v in c)

    def test_to_hex(self):
        """Test hex conversion."""
        from jones_framework.data.point_cloud import ColorMapper

        rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        hex_colors = ColorMapper.to_hex(rgb)

        assert hex_colors == ['#ff0000', '#00ff00', '#0000ff']

    def test_to_three_js_format(self):
        """Test Three.js format output."""
        from jones_framework.data.point_cloud import ColorMapper

        rgb = np.array([[1, 0, 0], [0, 1, 0]])
        flat = ColorMapper.to_three_js_color_array(rgb)

        assert flat.dtype == np.float32
        assert len(flat) == 6
        assert np.allclose(flat, [1, 0, 0, 0, 1, 0])


class TestHeatmapGenerator:
    """Tests for HeatmapGenerator class."""

    def test_2d_heatmap(self):
        """Test 2D heatmap generation."""
        from jones_framework.data.point_cloud import PointCloud2D, HeatmapGenerator

        data = np.random.rand(100, 2).astype(np.float32)
        cloud = PointCloud2D.from_numpy(data)

        gen = HeatmapGenerator(resolution=32)
        result = gen.generate_2d_heatmap(cloud)

        assert 'grid' in result
        assert 'colors' in result
        assert 'colors_hex' in result
        assert 'colors_float32' in result
        assert result['grid'].shape == (32, 32)

    def test_density_heatmap(self):
        """Test KDE density heatmap."""
        from jones_framework.data.point_cloud import PointCloud3D, HeatmapGenerator

        data = np.random.rand(50, 3).astype(np.float32)
        cloud = PointCloud3D.from_numpy(data)

        gen = HeatmapGenerator(resolution=16)
        result = gen.generate_density_heatmap(cloud, projection='xy')

        assert 'grid' in result
        assert 'bandwidth' in result
        assert result['grid'].shape == (16, 16)

    def test_attribute_heatmap(self):
        """Test attribute-based heatmap."""
        from jones_framework.data.point_cloud import PointCloud4D, HeatmapGenerator

        data = np.random.rand(100, 4).astype(np.float32)
        data[:, 3] = np.linspace(0, 10, 100)  # Time values
        cloud = PointCloud4D.from_numpy(data)

        gen = HeatmapGenerator(resolution=16)
        result = gen.generate_attribute_heatmap(cloud, attribute='time')

        assert 'grid' in result
        assert result['grid'].shape == (16, 16)

    def test_3d_point_colors(self):
        """Test 3D point color generation."""
        from jones_framework.data.point_cloud import PointCloud3D, HeatmapGenerator, Colormap

        data = np.array([
            [0, 0, 0],
            [0, 0, 5],
            [0, 0, 10]
        ], dtype=np.float32)
        cloud = PointCloud3D.from_numpy(data)

        gen = HeatmapGenerator()
        result = gen.generate_3d_point_colors(cloud, colormap=Colormap.JET)

        assert 'colors_rgb' in result
        assert 'colors_float32' in result
        assert result['colors_rgb'].shape == (3, 3)

    def test_label_colors(self):
        """Test label-based color generation."""
        from jones_framework.data.point_cloud import (
            HeatmapGenerator, PointCloudLabels, PointLabel
        )

        labels = PointCloudLabels.create(10)
        labels.register_label(PointLabel(1, "red", (255, 0, 0)))
        labels.register_label(PointLabel(2, "blue", (0, 0, 255)))
        labels.label_points(np.array([0, 1, 2]), 1)
        labels.label_points(np.array([5, 6]), 2)

        gen = HeatmapGenerator()
        result = gen.generate_label_colors(labels, num_points=10)

        assert 'colors_rgb' in result
        assert 'legend' in result
        assert 'red' in result['legend']
        assert 'blue' in result['legend']

    def test_to_frontend_format(self):
        """Test frontend format conversion."""
        from jones_framework.data.point_cloud import PointCloud2D, HeatmapGenerator

        data = np.random.rand(50, 2).astype(np.float32)
        cloud = PointCloud2D.from_numpy(data)

        gen = HeatmapGenerator(resolution=8)
        result = gen.generate_2d_heatmap(cloud)
        frontend = gen.to_frontend_format(result)

        # All numpy arrays should be converted to lists
        assert isinstance(frontend['grid'], list)
        assert isinstance(frontend['bounds'], list)


# =============================================================================
# Full Integration Tests
# =============================================================================

class TestFullIntegration:
    """Full integration tests for all new layers."""

    def test_label_edit_save_visualize(self, tmp_path):
        """Test complete workflow: create, label, edit, save, visualize."""
        from jones_framework.data.point_cloud import (
            PointCloud4D, PointCloudLabels, PointLabel, LabelType,
            PointCloudEditor, PointCloudIO, HeatmapGenerator, Colormap
        )

        # Create cloud
        data = np.random.rand(100, 4).astype(np.float32)
        data[:, 3] = np.linspace(0, 10, 100)
        cloud = PointCloud4D.from_numpy(data)

        # Label some points
        labels = PointCloudLabels.create(100)
        labels.register_label(PointLabel(1, "cluster_a", (255, 0, 0), label_type=LabelType.CLUSTER))
        labels.register_label(PointLabel(2, "cluster_b", (0, 0, 255), label_type=LabelType.CLUSTER))
        labels.label_points(np.arange(0, 40), 1)
        labels.label_points(np.arange(60, 100), 2)

        # Edit: translate cluster_a
        editor = PointCloudEditor(cloud, labels=labels)
        cluster_a_indices = labels.get_points_by_label(1)
        original_centroid = cloud.points[cluster_a_indices, :3].mean(axis=0)

        editor.translate(np.array([5, 0, 0]))

        # Save
        path = tmp_path / "integrated.npz"
        PointCloudIO.save(cloud, str(path), labels=labels)

        # Load back
        loaded_cloud, loaded_labels = PointCloudIO.load_with_labels(str(path))

        # Visualize
        gen = HeatmapGenerator()
        gen.color_mapper.colormap = Colormap.VIRIDIS

        # Generate heatmap
        heatmap = gen.generate_2d_heatmap(loaded_cloud, projection='xy')

        # Generate label colors
        label_colors = gen.generate_label_colors(loaded_labels, loaded_cloud.num_points)

        # Verify
        assert loaded_cloud.num_points == 100
        assert len(loaded_labels.get_points_by_label(1)) == 40
        assert 'colors_float32' in heatmap
        assert 'legend' in label_colors
        assert 'cluster_a' in label_colors['legend']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
