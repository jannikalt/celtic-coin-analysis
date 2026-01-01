import numpy as np
import cv2
from PIL import Image
import torch
import os
import time

class CoinMatcher:
    def __init__(self):
        self.debug_dir = "debug_matching"
        os.makedirs(self.debug_dir, exist_ok=True)

    def save_debug_image(self, name, image):
        """
        Saves an intermediate image for debugging.
        """
        # Temporary marked code for storing images of intermediate steps
        # timestamp = int(time.time() * 1000)
        # filename = f"{self.debug_dir}/{name}_{timestamp}.png"
        # cv2.imwrite(filename, image)


    def preprocess_mask(self, mask: np.ndarray, target_size=(128, 128), border: int = 0):
        """
        Centers the mask and resizes it to a fixed size.
        """
        # Ensure a clean binary mask (important for contour-based ops)
        mask_u8 = ((mask > 0).astype(np.uint8) * 255)

        # Find contours to get bounding box
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(target_size, dtype=np.uint8)
        
        # Get largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop
        cropped = mask_u8[y:y+h, x:x+w]
        
        # Resize maintaining aspect ratio
        h, w = cropped.shape
        target_h, target_w = target_size

        # Optional border: keep the resized object away from the image boundary.
        # This helps avoid edge artifacts when extracting a rim/outline.
        border = int(max(0, border))
        inner_h = max(1, target_h - 2 * border)
        inner_w = max(1, target_w - 2 * border)

        scale = min(inner_h / h, inner_w / w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Pad to target size (center it)
        result = np.zeros((target_h, target_w), dtype=np.uint8)
        y_off = border + (inner_h - new_h) // 2
        x_off = border + (inner_w - new_w) // 2
        result[y_off:y_off+new_h, x_off:x_off+new_w] = resized

        self.save_debug_image("masks", result)

        return result

    def extract_color_features(self, image: np.ndarray, mask: np.ndarray):
        """
        Extracts the average color of the coin using the mask.
        Returns a normalized RGB vector.
        """
        # Ensure mask is boolean
        mask_bool = mask > 0
        
        if not np.any(mask_bool):
            return np.zeros(3)
            
        # Calculate mean color of the masked area
        # Image is RGB (from PIL -> np.array)
        mean_color = cv2.mean(image, mask=mask.astype(np.uint8))[:3]
        
        # Normalize to 0-1
        return np.array(mean_color) / 255.0

    def calculate_color_similarity(self, color1, color2):
        """
        Calculates similarity between two color vectors.
        Uses 1 - Euclidean distance (normalized).
        """
        # Max distance in unit cube is sqrt(3) ~ 1.732
        dist = np.linalg.norm(color1 - color2)
        # Normalize distance to 0-1 range roughly
        # If dist is 0, similarity is 1. If dist is large, similarity drops.
        # Using a Gaussian kernel or simple linear mapping
        return max(0.0, 1.0 - (dist / 1.0)) # Assuming colors are somewhat close, scaling by 1.0

    def extract_edges(self, mask: np.ndarray, thickness: int = 5):
        """
        Extracts the outer edge/rim of the mask.

        Uses the external contour (largest component) and draws it as a thick outline.
        This avoids holes from interior structures and prevents border artifacts that
        can happen with large-kernel morphological gradients.
        """
        thickness = int(max(1, thickness))
        mask_u8 = ((mask > 0).astype(np.uint8) * 255)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(mask_u8)

        c = max(contours, key=cv2.contourArea)

        # Fill the *outer* contour, then erode and subtract to get an inside-only rim.
        # This keeps the outer boundary fixed (no outward growth) and avoids holes.
        filled = np.zeros_like(mask_u8)
        cv2.drawContours(filled, [c], contourIdx=-1, color=255, thickness=-1)

        k = max(3, 2 * thickness + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(filled, kernel, iterations=1)
        rim = cv2.subtract(filled, eroded)
        return rim

    def calculate_iou(self, mask1, mask2):
        """
        Calculates Intersection over Union.
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union
    
    def get_best_rotation_match(self, mask1, mask2, step=5):
        """
        Finds the best rotation and flip of mask2 to match mask1.
        Returns (best_iou, best_angle, best_flipped)
        """
        best_iou = -1.0
        best_angle = 0
        best_flipped = False
        
        rows, cols = mask2.shape
        center = (cols / 2, rows / 2)

        # Check normal and flipped (mirrored)
        candidates = [(mask2, False), (cv2.flip(mask2, 1), True)]

        for img, is_flipped in candidates:
            for angle in range(0, 360, step):
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(img, M, (cols, rows), flags=cv2.INTER_NEAREST)
                
                iou = self.calculate_iou(mask1, rotated)
                
                if iou > best_iou:
                    best_iou = iou
                    best_angle = angle
                    best_flipped = is_flipped
                
        return best_iou, best_angle, best_flipped

    def match_coins(self, masks: list[torch.Tensor] | list[np.ndarray], original_image: Image.Image, color_weight: float = 0.2):
        """
        Matches coins based on mask edge similarity and color similarity.
        Returns a list of matches: [{'indices': (i, j), 'score': float, 'edge_score': float, 'color_score': float, 'angle': int, 'flipped': bool}, ...]
        """
        # Convert original image to numpy for color extraction
        img_np = np.array(original_image)

        # Convert masks to numpy list and preprocess
        processed_masks = []
        color_features = []

        edge_thickness = 10

        for m in masks:
            if isinstance(m, torch.Tensor):
                m = m.cpu().numpy()
            m = (m * 255).astype(np.uint8)
            
            # Extract color features using full mask and full image
            # This avoids size mismatch issues
            color = self.extract_color_features(img_np, m)
            color_features.append(color)

            # Get centered, resized mask
            # Keep a small border so the outline/rim can't touch the image boundary.
            # If it does, edge extraction can result in holes.
            full_mask = self.preprocess_mask(m, border=1)
            # Extract edges for matching
            edge_mask = self.extract_edges(full_mask, thickness=edge_thickness)

            self.save_debug_image("edges", edge_mask)

            processed_masks.append(edge_mask)

        n = len(processed_masks)
        if n < 3:
            return []

        # Calculate all pairwise scores
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_score, angle, flipped = self.get_best_rotation_match(processed_masks[i], processed_masks[j])
                
                # Use pre-calculated color features
                color_score = self.calculate_color_similarity(color_features[i], color_features[j])

                # Weighted Score
                total_score = (1.0 - color_weight) * edge_score + color_weight * color_score

                pairs.append({
                    'indices': (i, j),
                    'score': total_score,
                    'edge_score': edge_score,
                    'color_score': color_score,
                    'angle': angle,
                    'flipped': flipped
                })

        # Sort by score descending
        pairs.sort(key=lambda x: x['score'], reverse=True)

        matches = []
        used_indices = set()

        for pair in pairs:
            i, j = pair['indices']
            if i not in used_indices and j not in used_indices:
                matches.append(pair)
                used_indices.add(i)
                used_indices.add(j)

        return matches