import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from hcaptcha_challenger.helper.create_coordinate_grid import FloatRect
from hcaptcha_challenger.helper.visualize_attention_points import (
    show_answer_points,
    create_comparison_view,
    _parse_answer_dict,
)
from hcaptcha_challenger.models import (
    ImageAreaSelectChallenge,
    ImageDragDropChallenge,
    PointCoordinate,
    SpatialPath,
)


@pytest.fixture
def test_image():
    """Create a simple test image."""
    return np.ones((431, 501, 3), dtype=np.uint8) * 255


@pytest.fixture
def bbox():
    """Create test bounding box."""
    return FloatRect(x=0, y=0, width=501, height=431)


@pytest.fixture
def area_select_answer():
    """Create sample area select answer."""
    return ImageAreaSelectChallenge(
        challenge_prompt="Select all images with traffic lights",
        points=[
            PointCoordinate(x=100, y=100),
            PointCoordinate(x=200, y=200),
            PointCoordinate(x=300, y=150),
        ],
    )


@pytest.fixture
def drag_drop_answer():
    """Create sample drag drop answer."""
    return ImageDragDropChallenge(
        challenge_prompt="Drag the pipe from the right to complete the puzzle",
        paths=[
            SpatialPath(
                start_point=PointCoordinate(x=400, y=200), end_point=PointCoordinate(x=250, y=250)
            )
        ],
    )


@pytest.fixture
def area_select_dict():
    """Create dictionary representation of area select answer."""
    return {
        "challenge_prompt": "Select all images with traffic lights",
        "points": [{"x": 100, "y": 100}, {"x": 200, "y": 200}, {"x": 300, "y": 150}],
    }


@pytest.fixture
def drag_drop_dict():
    """Create dictionary representation of drag drop answer."""
    return {
        "challenge_prompt": "Drag the pipe from the right to complete the puzzle",
        "paths": [{"start_point": {"x": 400, "y": 200}, "end_point": {"x": 250, "y": 250}}],
    }


class TestShowAnswerPoints:

    def test_parse_answer_dict_area_select(self, area_select_dict):
        """Test parsing area select answer from dictionary."""
        parsed = _parse_answer_dict(area_select_dict)
        assert isinstance(parsed, ImageAreaSelectChallenge)
        assert parsed.challenge_prompt == area_select_dict["challenge_prompt"]
        assert len(parsed.points) == 3
        assert parsed.points[0].x == 100
        assert parsed.points[0].y == 100

    def test_parse_answer_dict_drag_drop(self, drag_drop_dict):
        """Test parsing drag drop answer from dictionary."""
        parsed = _parse_answer_dict(drag_drop_dict)
        assert isinstance(parsed, ImageDragDropChallenge)
        assert parsed.challenge_prompt == drag_drop_dict["challenge_prompt"]
        assert len(parsed.paths) == 1
        assert parsed.paths[0].start_point.x == 400
        assert parsed.paths[0].end_point.x == 250

    def test_parse_answer_dict_invalid(self):
        """Test parsing invalid dictionary raises error."""
        invalid_dict = {"challenge_prompt": "Invalid", "invalid_key": []}
        with pytest.raises(ValueError):
            _parse_answer_dict(invalid_dict)

    @patch('matplotlib.pyplot.show')
    def test_show_answer_points_area_select(self, mock_show, test_image, area_select_answer, bbox):
        """Test visualizing area select answers."""
        result = show_answer_points(test_image, area_select_answer, bbox, show_plot=False)

        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3  # Should be 3D (height, width, channels)
        assert result.shape[2] == 3  # RGB channels

    @patch('matplotlib.pyplot.show')
    def test_show_answer_points_drag_drop(self, mock_show, test_image, drag_drop_answer, bbox):
        """Test visualizing drag drop answers."""
        result = show_answer_points(test_image, drag_drop_answer, bbox, show_plot=False)

        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        assert result.shape[2] == 3

    @patch('matplotlib.pyplot.show')
    def test_show_answer_points_with_dict(self, mock_show, test_image, area_select_dict, bbox):
        """Test visualizing answers from dictionary."""
        result = show_answer_points(test_image, area_select_dict, bbox, show_plot=False)

        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3

    @patch('matplotlib.pyplot.show')
    def test_show_answer_points_auto_bbox(self, mock_show, test_image, area_select_answer):
        """Test auto-detection of bbox when not provided."""
        result = show_answer_points(test_image, area_select_answer, bbox=None, show_plot=False)

        assert isinstance(result, np.ndarray)

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_show_answer_points_save(
        self, mock_savefig, mock_show, test_image, area_select_answer, bbox
    ):
        """Test saving visualization to file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            save_path = tmp.name

        try:
            result = show_answer_points(
                test_image, area_select_answer, bbox, save_path=save_path, show_plot=False
            )

            # Check that savefig was called
            mock_savefig.assert_called_once()

            # Check result is still returned
            assert isinstance(result, np.ndarray)
        finally:
            # Clean up temp file
            if os.path.exists(save_path):
                os.unlink(save_path)

    @patch('matplotlib.pyplot.show')
    def test_show_answer_points_custom_params(
        self, mock_show, test_image, area_select_answer, bbox
    ):
        """Test visualization with custom parameters."""
        result = show_answer_points(
            test_image,
            area_select_answer,
            bbox,
            show_plot=False,
            point_color='green',
            point_size=200,
            alpha=0.5,
        )

        assert isinstance(result, np.ndarray)

    @patch('matplotlib.pyplot.show')
    def test_create_comparison_view(self, mock_show, test_image, drag_drop_answer, bbox):
        """Test creating comparison view."""
        # Create a second test image
        coord_image = np.ones((431, 501, 3), dtype=np.uint8) * 200

        result = create_comparison_view(test_image, coord_image, drag_drop_answer, bbox)

        # Check that result is a numpy array
        assert isinstance(result, np.ndarray)
        assert len(result.shape) == 3
        # Comparison view should be wider (side-by-side)
        assert result.shape[1] > test_image.shape[1]

    def test_load_image_from_path(self, test_image, area_select_answer, bbox):
        """Test loading image from file path."""
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, test_image)

        try:
            with patch('matplotlib.pyplot.show'):
                result = show_answer_points(temp_path, area_select_answer, bbox, show_plot=False)

                assert isinstance(result, np.ndarray)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_invalid_image_path(self, area_select_answer, bbox):
        """Test error handling for invalid image path."""
        with pytest.raises(FileNotFoundError):
            show_answer_points("nonexistent_image.png", area_select_answer, bbox, show_plot=False)


class TestIntegrationWithRealImages:
    """Integration tests with real challenge images if available."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up paths to test images."""
        self.base_path = Path(__file__).parent / "challenge_view"
        self.drag_drop_path = self.base_path / "image_drag_drop"
        self.area_select_path = self.base_path / "image_label_area_select"
        self.bbox = FloatRect(x=0, y=0, width=501, height=431)

    @patch('matplotlib.pyplot.show')
    def test_with_real_drag_drop_image(self, mock_show):
        """Test with real drag drop image if available."""
        # Look for any PNG file in the drag drop directory
        if self.drag_drop_path.exists():
            png_files = list(self.drag_drop_path.glob("*.png"))
            if png_files and not png_files[0].name.startswith("coordinate_grid"):
                test_answer = {
                    "challenge_prompt": "Test drag drop challenge",
                    "paths": [
                        {"start_point": {"x": 450, "y": 200}, "end_point": {"x": 300, "y": 250}}
                    ],
                }

                result = show_answer_points(png_files[0], test_answer, self.bbox, show_plot=False)

                assert isinstance(result, np.ndarray)

    @patch('matplotlib.pyplot.show')
    def test_with_real_area_select_image(self, mock_show):
        """Test with real area select image if available."""
        if self.area_select_path.exists():
            png_files = list(self.area_select_path.glob("*.png"))
            if png_files and not png_files[0].name.startswith("coordinate_grid"):
                test_answer = {
                    "challenge_prompt": "Test area select challenge",
                    "points": [{"x": 150, "y": 150}, {"x": 350, "y": 280}],
                }

                result = show_answer_points(png_files[0], test_answer, self.bbox, show_plot=False)

                assert isinstance(result, np.ndarray)
