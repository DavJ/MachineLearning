"""
Biquaternion transformation module for advanced number prediction.

This module implements biquaternion-based transformations to enhance
the feature space for lottery number prediction. Biquaternions combine
quaternions with complex numbers for richer mathematical representations.
"""

import numpy as np
from typing import Tuple, List


class Biquaternion:
    """
    Represents a biquaternion: q = w + xi + yj + zk where w, x, y, z are complex numbers.
    """
    
    def __init__(self, w: complex, x: complex, y: complex, z: complex):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def __mul__(self, other: 'Biquaternion') -> 'Biquaternion':
        """Quaternion multiplication."""
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Biquaternion(w, x, y, z)
    
    def conjugate(self) -> 'Biquaternion':
        """Return the conjugate of the biquaternion."""
        return Biquaternion(self.w, -self.x, -self.y, -self.z)
    
    def norm(self) -> float:
        """Calculate the norm of the biquaternion."""
        return np.sqrt(
            abs(self.w)**2 + abs(self.x)**2 + 
            abs(self.y)**2 + abs(self.z)**2
        )
    
    def normalize(self) -> 'Biquaternion':
        """Return a normalized biquaternion."""
        n = self.norm()
        if n < 1e-10:
            return Biquaternion(1+0j, 0j, 0j, 0j)
        return Biquaternion(self.w/n, self.x/n, self.y/n, self.z/n)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [real(w), imag(w), real(x), imag(x), ...]."""
        return np.array([
            self.w.real, self.w.imag,
            self.x.real, self.x.imag,
            self.y.real, self.y.imag,
            self.z.real, self.z.imag
        ])


def numbers_to_biquaternion(numbers: List[int], max_number: int = 49) -> Biquaternion:
    """
    Convert a list of lottery numbers to a biquaternion representation.
    
    Args:
        numbers: List of lottery numbers (1-49)
        max_number: Maximum number in the lottery (default 49)
    
    Returns:
        Biquaternion representation of the numbers
    """
    # Normalize numbers to [0, 1]
    normalized = [n / max_number for n in numbers]
    
    # Create complex components from pairs of numbers
    # Pad with zeros if needed
    padded = normalized + [0.0] * (8 - len(normalized))
    
    w = complex(padded[0], padded[1])
    x = complex(padded[2], padded[3])
    y = complex(padded[4], padded[5])
    z = complex(padded[6], padded[7] if len(padded) > 7 else 0)
    
    return Biquaternion(w, x, y, z)


def biquaternion_transform(numbers: List[int]) -> np.ndarray:
    """
    Transform lottery numbers using biquaternion representation.
    
    This creates a richer feature space by:
    1. Converting numbers to biquaternion
    2. Applying rotation transformations
    3. Extracting meaningful features
    
    Args:
        numbers: List of lottery numbers
    
    Returns:
        Transformed feature vector
    """
    bq = numbers_to_biquaternion(numbers)
    
    # Apply rotation transformations
    # Rotate around different axes
    rotation1 = Biquaternion(
        complex(np.cos(np.pi/4), 0),
        complex(np.sin(np.pi/4), 0),
        0j, 0j
    ).normalize()
    
    rotation2 = Biquaternion(
        complex(np.cos(np.pi/6), 0),
        0j,
        complex(np.sin(np.pi/6), 0),
        0j
    ).normalize()
    
    # Apply rotations
    rotated1 = rotation1 * bq
    rotated2 = rotation2 * bq
    
    # Combine original and rotated features
    features = np.concatenate([
        bq.to_array(),
        rotated1.to_array(),
        rotated2.to_array(),
    ])
    
    return features


def theta_orthogonalization(features: np.ndarray, theta: float = np.pi/4) -> np.ndarray:
    """
    Apply theta-based orthogonalization to features.
    
    This creates an orthogonal basis rotated by theta angle,
    reducing correlation between features.
    
    Args:
        features: Input feature vector
        theta: Rotation angle (default π/4)
    
    Returns:
        Orthogonalized feature vector
    """
    n = len(features)
    
    # Create rotation matrix
    rotation_matrix = np.eye(n)
    for i in range(0, n-1, 2):
        if i+1 < n:
            c, s = np.cos(theta), np.sin(theta)
            rotation_matrix[i:i+2, i:i+2] = np.array([[c, -s], [s, c]])
    
    return rotation_matrix @ features


def apply_biquaternion_theta_transform(numbers: List[int]) -> np.ndarray:
    """
    Apply complete biquaternion and theta transformation pipeline.
    
    Args:
        numbers: List of lottery numbers
    
    Returns:
        Fully transformed feature vector
    """
    # Apply biquaternion transformation
    bq_features = biquaternion_transform(numbers)
    
    # Apply theta orthogonalization
    ortho_features = theta_orthogonalization(bq_features)
    
    return ortho_features


def probability_to_biquaternion_features(probability_vector: np.ndarray) -> np.ndarray:
    """
    Convert probability vector (49 numbers) to biquaternion features.
    
    Args:
        probability_vector: Probability distribution over 49 numbers
    
    Returns:
        Biquaternion-transformed features
    """
    # Extract top numbers with their probabilities
    top_indices = np.argsort(probability_vector)[-7:][::-1]
    top_numbers = (top_indices + 1).tolist()
    
    return apply_biquaternion_theta_transform(top_numbers)
