## diagram_processor.py
import os
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
from skimage import feature, transform, color
import math

# Configuration parameters and constants (can be extended or loaded from config.yaml)
# For demonstration, using fixed parameters; in production, load from config if needed.
EDGE_DETECTION_SIGMA: float = 1.0
HOUGH_LINE_THRESHOLD: float = 0.3
HOUGH_MIN_LINE_LENGTH: int = 50
HOUGH_LINE_GAP: int = 10
CIRCLE_DETECTION_RADIUS_RANGE: tuple = (20, 100)  # Adjust based on expected circle size
TOLERANCE: float = 5.0  # Pixel tolerance for geometric relations

class DiagramProcessor:
    def __init__(self, image_path: str):
        self.image_path: str = image_path
        self.image: Optional[np.ndarray] = None
        self.gray_image: Optional[np.ndarray] = None
        self.edges: Optional[np.ndarray] = None
        self.lines: List[Dict[str, Any]] = []
        self.circles: List[Dict[str, Any]] = []
        self.points: List[Dict[str, Any]] = []
        # Relations
        self.collinearity: List[List[int]] = []
        self.on_line_relations: List[Dict[str, int]] = []
        self.on_circle_relations: List[Dict[str, int]] = []
        self.parallels: List[Dict[str, int]] = []
        self.intersections: List[Dict[str, int]] = []

    def load_image(self) -> None:
        """Loads the image file."""
        img = Image.open(self.image_path)
        self.image = np.array(img)
        print(f"Loaded image with shape {self.image.shape}")

    def preprocess_image(self) -> None:
        """Converts to grayscale and extracts edges."""
        if self.image is None:
            raise RuntimeError("Image not loaded.")
        # Convert to grayscale
        if len(self.image.shape) == 3:
            gray = color.rgb2gray(self.image)
        else:
            gray = self.image / 255.0
        self.gray_image = gray
        # Edge detection
        self.edges = feature.canny(
            gray, sigma=EDGE_DETECTION_SIGMA
        )

    def detect_lines(self) -> None:
        """Detects straight lines using probabilistic Hough transform."""
        if self.edges is None:
            raise RuntimeError("Edge image not processed.")
        lines = transform.probabilistic_hough_line(
            self.edges,
            threshold=int(HOUGH_LINE_THRESHOLD * np.max(self.edges)),
            line_length=HOUGH_MIN_LINE_LENGTH,
            line_gap=HOUGH_LINE_GAP
        )
        for idx, (p0, p1) in enumerate(lines):
            line_dict = {
                'id': f'line_{idx}',
                'start_point': p0,
                'end_point': p1,
                'length': np.linalg.norm(np.array(p0) - np.array(p1)),
                'direction_vector': np.array(p1) - np.array(p0)
            }
            self.lines.append(line_dict)
        print(f"Detected {len(self.lines)} lines.")

    def detect_circles(self) -> None:
        """Detect circles using Hough Circle Transform."""
        if self.gray_image is None:
            raise RuntimeError("Grayscale image not processed.")
        # Using Hough circle detection in skimage
        # Parameters may need tuning based on image resolution
        # Convert to uint8 if needed
        from skimage.transform import hough_circle, hough_circle_peaks
        # Estimate radius range based on image size
        min_radius, max_radius = CIRCLE_DETECTION_RADIUS_RANGE
        # Edge detection for circle detection
        edges = self.edges
        # For better detection, can adjust number of radii
        radii = np.arange(min_radius, max_radius, 2)
        hough_res = hough_circle(edges, radii)
        accums, cx, cy, radii_detected = hough_circle_peaks(
            hough_res, radii, total_num_peaks=10
        )
        for idx, (x, y, r) in enumerate(zip(cx, cy, radii_detected)):
            circle_dict = {
                'id': f'circle_{idx}',
                'center': (float(x), float(y)),
                'radius': float(r)
            }
            self.circles.append(circle_dict)
        print(f"Detected {len(self.circles)} circles.")

    def extract_points(self) -> None:
        """Extract points from detected line endpoints and circle centers."""
        point_coords = {}
        point_id_counter = 0
        # Collect endpoints of lines
        for line in self.lines:
            for pt in [line['start_point'], line['end_point']]:
                key = (round(pt[0]), round(pt[1]))
                if key not in point_coords:
                    point_id = f'point_{point_id_counter}'
                    point_coords[key] = {'id': point_id, 'coord': (float(pt[0]), float(pt[1]))}
                    point_id_counter += 1
        # Collect circle centers
        for circle in self.circles:
            center_unrounded = circle['center']
            key = (round(center_unrounded[0]), round(center_unrounded[1]))
            if key not in point_coords:
                point_id = f'point_{point_id_counter}'
                point_coords[key] = {'id': point_id, 'coord': (float(center_unrounded[0]), float(center_unrounded[1]))}
                point_id_counter += 1
        # Convert to list
        self.points = list(point_coords.values())

    def check_collinearity(self) -> None:
        """Determine collinear points by line detection."""
        # Build a list of point coordinates for distance computation
        coords = {p['id']: p['coord'] for p in self.points}
        point_ids = list(coords.keys())
        for i in range(len(point_ids)):
            for j in range(i + 1, len(point_ids)):
                p_id1 = point_ids[i]
                p_id2 = point_ids[j]
                p1 = np.array(coords[p_id1])
                p2 = np.array(coords[p_id2])
                for k in range(j + 1, len(point_ids)):
                    p_id3 = point_ids[k]
                    p3 = np.array(coords[p_id3])
                    if self._are_collinear(p1, p2, p3):
                        self.collinearity.append([p_id1, p_id2, p_id3])
        print(f"Found {len(self.collinearity)} collinear triplets.")

    def _are_collinear(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> bool:
        """Check if three points are collinear within tolerance."""
        area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
        return area < TOLERANCE

    def relate_points_on_lines(self) -> None:
        """Determine if points lie on detected lines."""
        for point in self.points:
            point_id = point['id']
            point_coord = np.array(point['coord'])
            for line in self.lines:
                start = np.array(line['start_point'])
                end = np.array(line['end_point'])
                if self._point_on_line(point_coord, start, end):
                    relation = {'point_id': point_id, 'line_id': line['id']}
                    self.on_line_relations.append(relation)
        print(f"Established {len(self.on_line_relations)} point-on-line relations.")

    def _point_on_line(self, pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> bool:
        """Determine if point is on the line segment within tolerance."""
        line_vec = end - start
        point_vec = pt - start
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(pt - start) < TOLERANCE
        # Project point onto line
        projection = np.dot(point_vec, line_vec) / line_len
        if 0 - TOLERANCE <= projection <= line_len + TOLERANCE:
            # Check perpendicular distance
            closest_point = start + (projection / line_len) * line_vec
            dist = np.linalg.norm(pt - closest_point)
            return dist < TOLERANCE
        return False

    def relate_points_on_circles(self) -> None:
        """Determine points lying on circles."""
        for point in self.points:
            point_id = point['id']
            pt = np.array(point['coord'])
            for circle in self.circles:
                center = np.array(circle['center'])
                radius = circle['radius']
                dist = np.linalg.norm(pt - center)
                if abs(dist - radius) < TOLERANCE:
                    relation = {'point_id': point_id, 'circle_center': circle['id']}
                    self.on_circle_relations.append(relation)
        print(f"Established {len(self.on_circle_relations)} point-on-circle relations.")

    def compute_circle_intersections(self) -> None:
        """Compute intersection points between pairs of circles."""
        circle_pairs = []
        for i in range(len(self.circles)):
            for j in range(i + 1, len(self.circles)):
                circle_pairs.append((self.circles[i], self.circles[j]))
        for c1, c2 in circle_pairs:
            pts = self._circle_circle_intersections(c1['center'], c1['radius'], c2['center'], c2['radius'])
            for pt in pts:
                # Register as a point if not duplicated
                key = (round(pt[0]), round(pt[1]))
                point_id = f'intersect_{c1["id"]}_{c2["id"]}_{key}'
                self.points.append({'id': point_id, 'coord': (float(pt[0]), float(pt[1]))})
                self.intersections.append({'point_id': point_id, 'circles': (c1['id'], c2['id'])})
        print(f"Computed {len(self.intersections)} circle-circle intersection points.")

    def _circle_circle_intersections(self, c1: tuple, r1: float, c2: tuple, r2: float) -> List[np.ndarray]:
        """Calculate intersection points of two circles."""
        d = np.linalg.norm(np.array(c2) - np.array(c1))
        if d > r1 + r2 + TOLERANCE or d < abs(r1 - r2) - TOLERANCE:
            return []  # No intersection
        # Compute intersection points
        a = (r1**2 - r2**2 + d**2) / (2 * d)
        h_sq = r1**2 - a**2
        if h_sq < 0:
            return []
        h = math.sqrt(h_sq)
        mid_point = np.array(c1) + a / d * (np.array(c2) - np.array(c1))
        offset = h / d * np.array([-(c2[1] - c1[1]), c2[0] - c1[0]])
        intersection1 = mid_point + offset
        intersection2 = mid_point - offset
        if np.linalg.norm(intersection1 - intersection2) < TOLERANCE:
            return [intersection1]
        else:
            return [intersection1, intersection2]

    def determine_parallel_lines(self) -> None:
        """Estimate parallel lines based on their slopes."""
        for i in range(len(self.lines)):
            for j in range(i + 1, len(self.lines)):
                line1 = self.lines[i]
                line2 = self.lines[j]
                if self._are_lines_parallel(line1, line2):
                    self.parallels.append({'line1': line1['id'], 'line2': line2['id']})
        print(f"Found {len(self.parallels)} pairs of parallel lines.")

    def _are_lines_parallel(self, line1: Dict[str, Any], line2: Dict[str, Any]) -> bool:
        """Determine if two lines are parallel within a tolerance."""
        vec1 = line1['direction_vector']
        vec2 = line2['direction_vector']
        cross = np.cross(vec1, vec2)
        return abs(cross) < TOLERANCE

    def extract_relations(self) -> Dict[str, Any]:
        """Compile all extracted relations into structured dict."""
        relations = {
            'collinearity': self.collinearity,
            'on_line': self.on_line_relations,
            'on_circle': self.on_circle_relations,
            'parallel_lines': self.parallels,
            'intersections': self.intersections
        }
        return relations

    def process(self) -> Dict[str, Any]:
        """Complete pipeline: load, preprocess, detect primitives, and extract relations."""
        self.load_image()
        self.preprocess_image()
        self.detect_lines()
        self.detect_circles()
        self.extract_points()
        self.check_collinearity()
        self.relate_points_on_lines()
        self.relate_points_on_circles()
        self.compute_circle_intersections()
        self.determine_parallel_lines()
        relations = self.extract_relations()
        return {
            'points': self.points,
            'lines': self.lines,
            'circles': self.circles,
            'relations': relations
        }
