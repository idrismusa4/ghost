from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class _FaceLandmarks:
    """Container for the landmark output returned by :class:`FaceAnalysis`."""

    points: np.ndarray

    @classmethod
    def from_face(cls, face) -> "_FaceLandmarks":  # type: ignore[type-arg]
        if not hasattr(face, "landmark_2d_106"):
            raise ValueError("InsightFace model did not return 106-point landmarks.")
        return cls(points=np.asarray(face.landmark_2d_106, dtype=np.float32))


class Handler:
    """Lightweight replacement for the original MXNet-based landmark handler.

    The original implementation depended on MXNet 1.x which no longer provides
    wheels compatible with modern Python versions.  This reimplementation relies
    on :mod:`insightface.app.FaceAnalysis`, which offers the same 106-point
    facial landmark predictions while using ONNX Runtime under the hood.  The
    public methods are kept compatible with the previous handler so that the
    rest of the inference pipeline can run unchanged.
    """

    def __init__(
        self,
        prefix: str,
        epoch: int,
        im_size: int = 192,
        det_size: int = 224,
        ctx_id: int = 0,
        root: str = "./insightface_func/models",
    ) -> None:
        del prefix  # Unused with the InsightFace implementation.
        del epoch
        self._det_size = (det_size, det_size) if isinstance(det_size, int) else det_size
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else [
            "CPUExecutionProvider"
        ]
        self._app = FaceAnalysis(name="buffalo_l", root=root, providers=providers)
        # FaceAnalysis expects -1 for CPU execution.
        self._app.prepare(ctx_id=ctx_id, det_size=self._det_size)

    def _select_face(self, faces: Iterable[object]) -> _FaceLandmarks:
        best_face = None
        best_score: float = float("-inf")
        for face in faces:
            score = getattr(face, "det_score", 0.0) or 0.0
            if score > best_score:
                best_face = face
                best_score = score
        if best_face is None:
            raise ValueError("No face detected in the provided image.")
        return _FaceLandmarks.from_face(best_face)

    def _predict_landmarks(self, image: np.ndarray) -> np.ndarray:
        faces = self._app.get(image)
        if not faces:
            raise ValueError("No face detected in the provided image.")
        return self._select_face(faces).points

    def get_without_detection_without_transform(self, img: np.ndarray) -> np.ndarray:
        """Return 106-point landmarks for an already cropped face image."""

        return self._predict_landmarks(img)

    # Backwards-compatible aliases -------------------------------------------------
    def get_without_detection(self, img: np.ndarray, get_all: bool = False) -> np.ndarray:
        landmarks = self._predict_landmarks(img)
        if get_all:
            return np.expand_dims(landmarks, axis=0)
        return landmarks

    def get_without_detection_batch(
        self, img: np.ndarray, M: np.ndarray, IM: np.ndarray
    ) -> np.ndarray:
        # The batching API is unused in the current inference pipeline.  It is
        # provided for completeness and to maintain the previous signature.
        del M, IM
        landmarks = self._predict_landmarks(img)
        return np.expand_dims(landmarks, axis=0)
