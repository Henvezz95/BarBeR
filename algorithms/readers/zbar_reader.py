from pyzbar import pyzbar
import numpy as np
from time import perf_counter_ns
import cv2
from typing import List, Tuple, Dict, Any

def get_poly(decoding):
    poly = decoding.polygon
    return np.array([[p.x,p.y] for p in poly]).astype('int32')

def get_string(decoding):
    return decoding.data.decode('utf-8')

class ZbarReader:
    def __init__(self, 
                 pre_localizer: Any = None, 
                 padding: int = 20, 
                 down_ratio: float = 4.0, 
                 scales: List[float] = None, 
                 optimize: bool = True) -> None:
        """
        Initialize ZbarReader.
        Args:
            pre_localizer: Model for pre-localizing barcode regions.
            padding: Padding around detected boxes.
            down_ratio: Downscale factor for pre-localization.
            scales: List of scales for multi-resolution decoding.
            optimize: If True, stop trying additional scales once a decode is found.
        """
        if scales is None:
            scales = [0.25, 1.0]
        self.pre_localizer = pre_localizer
        self.padding = padding
        self.down_ratio = down_ratio
        self.scales = scales
        self.optimize = optimize
        self.timing = -1

    def _calculate_crop_coords(self, box: Tuple[int, int, int, int], image_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        """
        Compute crop coordinates in the original image.
        Args:
            box: (x, y, w, h) from pre-localizer.
            image_shape: (H, W, channels) of the image.
        Returns:
            (x0, y0, x1, y1)
        """
        H, W, _ = image_shape
        x, y, w, h = box
        x0 = int(max(0, self.down_ratio * x - self.padding))
        y0 = int(max(0, self.down_ratio * y - self.padding))
        x1 = int(min(self.down_ratio * (x + w) + self.padding, W))
        y1 = int(min(self.down_ratio * (y + h) + self.padding, H))
        return x0, y0, x1, y1

    def _multi_resolution_decode(self, image: np.ndarray, offset_x: int, offset_y: int) -> List[Dict[str, Any]]:
        """
        Try decoding at multiple resolutions.
        Args:
            image: Image crop to decode.
            offset_x: X offset of crop in original image.
            offset_y: Y offset of crop in original image.
        Returns:
            List of barcode dicts with keys 'box' and 'string'.
        """
        results = []
        # If optimizing, sort scales in ascending order (lowest resolution first)
        scales = sorted(self.scales) if self.optimize else self.scales
        for scale in scales:
            image_scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) if scale != 1.0 else image
            if decoded := pyzbar.decode(image_scaled):
                for res in decoded:
                    x, y, w, h = res.rect
                    # Map coordinates from scaled crop to original crop then add offset.
                    x_orig = int(x / scale) + offset_x
                    y_orig = int(y / scale) + offset_y
                    w_orig = int(w / scale)
                    h_orig = int(h / scale)
                    results.append({'box': [x_orig, y_orig, w_orig, h_orig], 'string': get_string(res)})
                if self.optimize:
                    break
        return results

    def _decode_with_prelocalization(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode barcodes using pre-localization and multi-resolution decoding.
        Args:
            image: The original image.
        Returns:
            List of barcode dicts.
        """
        # Downscale image for pre-localization.
        image_downscaled = cv2.resize(image, (0, 0),
                                      fx=1/self.down_ratio, fy=1/self.down_ratio,
                                      interpolation=cv2.INTER_AREA)
        boxes, _, _ = self.pre_localizer.detect(image_downscaled)
        start_time = perf_counter_ns()
        decoded_results = []
        for box in boxes:
            x0, y0, x1, y1 = self._calculate_crop_coords(box, image.shape)
            if (x1 - x0) > self.padding and (y1 - y0) > self.padding:
                crop = image[y0:y1, x0:x1]
                crop_results = self._multi_resolution_decode(crop, x0, y0)
                decoded_results.extend(crop_results)
        elapsed_time = (perf_counter_ns() - start_time) / 1e6
        self.timing = elapsed_time + self.pre_localizer.get_timing()
        return decoded_results

    def decode(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode barcodes from the image using multi-resolution decoding.
        Args:
            image: The input image.
        Returns:
            List of barcode dicts.
        """
        if self.pre_localizer is not None:
            return self._decode_with_prelocalization(image)
        start_time = perf_counter_ns()
        results = self._multi_resolution_decode(image, 0, 0)
        self.timing = (perf_counter_ns() - start_time) / 1e6
        return results

    def get_timing(self) -> float:
        """Return the last timing (ms)."""
        return self.timing
    

class ZbarReaderWithSegmentation:
    def __init__(self, pre_localizer: Any, padding: int = 20, min_area: int = 300, down_ratio: float = 4.0) -> None:
        """
        Args:
            pre_localizer: Segmentation model returning a tensor (H,W,2) with binary values.
            padding: Extra pixels to add around each detected region.
            min_area: Ignore small contours.
            down_ratio: Factor to downscale image for segmentation.
        """
        self.pre_localizer = pre_localizer
        self.padding = padding
        self.min_area = min_area
        self.down_ratio = down_ratio
        self.timing = -1

    def _extract_rotated_boxes_from_mask(self, mask: np.ndarray) -> List[Tuple]:
        """
        Extract rotated boxes from a binary mask.
        Args:
            mask: Binary image (0 or 255).
        Returns:
            List of rotated rectangles ((center), (w, h), angle).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            box = cv2.minAreaRect(cnt)
            boxes.append(box)
        return boxes

    def decode(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode barcodes using segmentation-based rotated box extraction.
        Args:
            image: Original image.
        Returns:
            List of dicts with 'box' (4 corner points) and 'string' (decoded barcode).
        """
        start_time = perf_counter_ns()

        # Downscale for segmentation if needed.
        if self.down_ratio != 1.0:
            image_downscaled = cv2.resize(image, (0, 0),
                                          fx=1/self.down_ratio, fy=1/self.down_ratio,
                                          interpolation=cv2.INTER_AREA)
        else:
            image_downscaled = image

        # Get segmentation (binary tensor: H,W,2)
        segmentation = self.pre_localizer.segment(image_downscaled)
        c,H,W = segmentation.shape
        results = []

        # Process both channels (for 1D and 2D barcodes)
        for ch in range(c):
            mask = np.uint8(segmentation[ch] * 255)
            boxes = self._extract_rotated_boxes_from_mask(mask)
            for box in boxes:
                center_ds, (w_ds, h_ds), angle = box  # In downscaled coordinates.
                # Scale coordinates to original resolution.
                center_orig = (center_ds[0] * self.down_ratio, center_ds[1] * self.down_ratio)
                w_orig = w_ds * self.down_ratio
                h_orig = h_ds * self.down_ratio
                # Add padding (in original resolution).
                w_pad = w_orig + 2 * self.padding
                h_pad = h_orig + 2 * self.padding
                # Rotation correction on original image.
                M = cv2.getRotationMatrix2D(center_orig, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
                corrected = cv2.getRectSubPix(rotated, (int(w_pad), int(h_pad)), center_orig)
                decoded_barcodes = pyzbar.decode(corrected)
                # Get padded rotated rectangle and convert to axis-aligned box
                padded_rect = (center_orig, (w_orig + 2 * self.padding, h_orig + 2 * self.padding), angle)
                pts = cv2.boxPoints(padded_rect)
                x, y, w, h = cv2.boundingRect(np.int0(pts))
                results.extend(
                    {'box': [x, y, w, h], 'string': get_string(d)}
                    for d in decoded_barcodes
                )
        self.timing = (perf_counter_ns() - start_time) / 1e6
        return results

    def get_timing(self) -> float:
        """Return the last timing (ms)."""
        return self.timing