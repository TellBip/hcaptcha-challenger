from pathlib import Path
from typing import Union, Optional, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from ..models import ImageAreaSelectChallenge, ImageDragDropChallenge, PointCoordinate, SpatialPath
from .create_coordinate_grid import FloatRect


def show_answer_points(
    image: Union[str, np.ndarray, Path],
    answer: Union[ImageAreaSelectChallenge, ImageDragDropChallenge, Dict[str, Any]],
    bbox: Optional[FloatRect] = None,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    **kwargs,
) -> np.ndarray:
    """
    Visualize CAPTCHA answer points on the image.

    Args:
        image: Input image (path or numpy array)
        answer: Answer object containing points or paths
        bbox: Bounding box for coordinate alignment (default: auto-detect from image)
        save_path: Optional path to save the visualization
        show_plot: Whether to display the plot (default: True)
        **kwargs: Additional visualization parameters:
            - point_color: Color for area select points (default: 'red')
            - point_size: Size of area select points (default: 100)
            - path_color: Color for drag paths (default: 'blue')
            - arrow_width: Width of drag arrows (default: 3)
            - alpha: Transparency of overlays (default: 0.7)

    Returns:
        Processed image with answer visualization
    """
    # Load image if path is provided
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(f"Could not load image from {image}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()

    # Auto-detect bbox from image if not provided
    if bbox is None:
        height, width = img.shape[:2]
        bbox = FloatRect(x=0, y=0, width=float(width), height=float(height))

    # Parse answer if it's a dictionary
    if isinstance(answer, dict):
        answer = _parse_answer_dict(answer)

    # Extract visualization parameters
    point_color = kwargs.get('point_color', 'red')
    point_size = kwargs.get('point_size', 100)
    path_color = kwargs.get('path_color', 'blue')
    arrow_width = kwargs.get('arrow_width', 3)
    alpha = kwargs.get('alpha', 0.7)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Display image
    x, y = bbox['x'], bbox['y']
    width, height = bbox['width'], bbox['height']
    ax.imshow(img, extent=(x, x + width, y + height, y))

    # Set axis limits and labels
    ax.set_xlim(x, x + width)
    ax.set_ylim(y + height, y)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)

    # Add title with challenge prompt
    title = 'Answer Visualization'
    if hasattr(answer, 'challenge_prompt'):
        title = f"{title}: {answer.challenge_prompt[:80]}..."
    ax.set_title(title, fontsize=14, pad=20)

    # Visualize based on answer type
    if isinstance(answer, ImageAreaSelectChallenge):
        _visualize_area_select(ax, answer, point_color, point_size, alpha)
    elif isinstance(answer, ImageDragDropChallenge):
        _visualize_drag_drop(ax, answer, path_color, arrow_width, alpha)

    # Add grid for reference
    ax.grid(True, color='gray', alpha=0.2, linestyle='--', linewidth=0.5)

    # Tight layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show plot if requested
    if show_plot:
        plt.show()

    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()  # type: ignore[arg-type]
    result_img = np.frombuffer(buf, dtype=np.uint8)
    result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)

    plt.close(fig)

    return result_img


def _parse_answer_dict(
    answer_dict: Dict[str, Any],
) -> Union[ImageAreaSelectChallenge, ImageDragDropChallenge]:
    """Parse dictionary answer into proper model object."""
    challenge_prompt = answer_dict.get('challenge_prompt', '')

    if 'points' in answer_dict:
        points = [PointCoordinate(x=p['x'], y=p['y']) for p in answer_dict['points']]
        return ImageAreaSelectChallenge(challenge_prompt=challenge_prompt, points=points)
    elif 'paths' in answer_dict:
        paths = [
            SpatialPath(
                start_point=PointCoordinate(x=p['start_point']['x'], y=p['start_point']['y']),
                end_point=PointCoordinate(x=p['end_point']['x'], y=p['end_point']['y']),
            )
            for p in answer_dict['paths']
        ]
        return ImageDragDropChallenge(challenge_prompt=challenge_prompt, paths=paths)
    else:
        raise ValueError("Answer dictionary must contain either 'points' or 'paths'")


def _visualize_area_select(
    ax: plt.Axes, answer: ImageAreaSelectChallenge, color: str, size: int, alpha: float
) -> None:
    """Visualize area select points."""
    for i, point in enumerate(answer.points, 1):
        # Draw point
        circle = Circle((point.x, point.y), radius=size / 10, color=color, alpha=alpha, zorder=5)
        ax.add_patch(circle)

        # Add label
        ax.annotate(
            f'P{i}',
            (point.x, point.y),
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=6,
        )

        # Add coordinate text
        ax.text(
            point.x,
            point.y - size / 5,
            f'({point.x}, {point.y})',
            color=color,
            fontsize=8,
            ha='center',
            va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )


def _visualize_drag_drop(
    ax: plt.Axes, answer: ImageDragDropChallenge, color: str, width: float, alpha: float
) -> None:
    """Visualize drag and drop paths."""
    for i, path in enumerate(answer.paths, 1):
        start = path.start_point
        end = path.end_point

        # Calculate arrow properties
        dx = end.x - start.x
        dy = end.y - start.y

        # Draw arrow
        arrow = FancyArrow(
            start.x,
            start.y,
            dx,
            dy,
            width=width,
            head_width=width * 4,
            head_length=width * 3,
            color=color,
            alpha=alpha,
            zorder=5,
            length_includes_head=True,
        )
        ax.add_patch(arrow)

        # Draw start point
        start_circle = Circle(
            (start.x, start.y), radius=width * 2, color='green', alpha=alpha, zorder=6
        )
        ax.add_patch(start_circle)
        ax.annotate(
            f'S{i}',
            (start.x, start.y),
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=7,
        )

        # Draw end point
        end_circle = Circle((end.x, end.y), radius=width * 2, color='red', alpha=alpha, zorder=6)
        ax.add_patch(end_circle)
        ax.annotate(
            f'E{i}',
            (end.x, end.y),
            color='white',
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='center',
            zorder=7,
        )

        # Add path label
        mid_x = (start.x + end.x) / 2
        mid_y = (start.y + end.y) / 2
        ax.text(
            mid_x,
            mid_y,
            f'Path {i}',
            color=color,
            fontsize=9,
            ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
            zorder=1000,
        )


def create_comparison_view(
    original_image: Union[str, np.ndarray, Path],
    coordinate_image: Union[str, np.ndarray, Path],
    answer: Union[ImageAreaSelectChallenge, ImageDragDropChallenge, Dict[str, Any]],
    bbox: Optional[FloatRect] = None,
    save_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Create a side-by-side comparison view of original and coordinate grid with answers.

    Args:
        original_image: Original challenge image
        coordinate_image: Image with coordinate grid
        answer: Answer object containing points or paths
        bbox: Bounding box for coordinate alignment
        save_path: Optional path to save the visualization
        **kwargs: Additional visualization parameters

    Returns:
        Combined comparison image
    """
    # Load images
    if isinstance(original_image, (str, Path)):
        orig_img = cv2.imread(str(original_image))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    else:
        orig_img = original_image.copy()

    if isinstance(coordinate_image, (str, Path)):
        coord_img = cv2.imread(str(coordinate_image))
        coord_img = cv2.cvtColor(coord_img, cv2.COLOR_BGR2RGB)
    else:
        coord_img = coordinate_image.copy()

    # Auto-detect bbox
    if bbox is None:
        height, width = orig_img.shape[:2]
        bbox = FloatRect(x=0, y=0, width=float(width), height=float(height))

    # Parse answer if needed
    if isinstance(answer, dict):
        answer = _parse_answer_dict(answer)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display original with answer
    x, y = bbox['x'], bbox['y']
    width, height = bbox['width'], bbox['height']

    ax1.imshow(orig_img, extent=(x, x + width, y + height, y))
    ax1.set_xlim(x, x + width)
    ax1.set_ylim(y + height, y)
    ax1.set_title('Challenge View', fontsize=14)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')

    # Display coordinate grid with answer
    ax2.imshow(coord_img, extent=(x, x + width, y + height, y))
    ax2.set_xlim(x, x + width)
    ax2.set_ylim(y + height, y)
    ax2.set_title('Spatial Helper View', fontsize=14)
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')

    # Add answers to both views
    for ax in [ax1, ax2]:
        if isinstance(answer, ImageAreaSelectChallenge):
            _visualize_area_select(ax, answer, 'red', 100, 0.7)
        elif isinstance(answer, ImageDragDropChallenge):
            _visualize_drag_drop(ax, answer, 'blue', 3, 0.7)

    # Add main title
    if hasattr(answer, 'challenge_prompt'):
        fig.suptitle(f"Challenge: {answer.challenge_prompt}", fontsize=16, y=1.02)

    plt.tight_layout()

    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # Show plot
    plt.show()

    # Convert to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()  # type: ignore[arg-type]
    result_img = np.frombuffer(buf, dtype=np.uint8)
    result_img = result_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)

    plt.close(fig)

    return result_img
