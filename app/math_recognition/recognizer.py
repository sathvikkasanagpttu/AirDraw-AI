"""
AirDraw AI - Handwritten Math & Arithmetic Recognition
======================================================
Recognizes handwritten mathematical expressions (+, -, x, /, =, digits, decimals)
drawn in the air and safely evaluates them using an AST parser.
"""

import ast
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.utils.logger import get_logger

logger = get_logger("MathRecognizer")

try:
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


@dataclass
class MathRecognitionResult:
    """Evaluation result for handwritten math expression."""
    symbols: List[str]
    boxes: List[Tuple[int, int, int, int]]
    text: str
    value: Optional[str]
    shown: str
    bbox: Tuple[int, int, int, int]


class MathRecognizer:
    """
    Parses ink bitmaps into recognized mathematical symbols and safely computes results.
    """

    FONTS = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_PLAIN,
    ]

    def __init__(self, lazy_train: bool = True):
        self.ready: bool = False
        self.error: Optional[str] = None
        self.clf: Optional[Any] = None
        self._is_trained: bool = False

        if not HAS_SKLEARN:
            self.error = "scikit-learn is required for handwriting recognition."
            logger.warning(self.error)
            return

        self.ready = True
        if not lazy_train:
            self.ensure_trained()

    def ensure_trained(self) -> bool:
        """Trains the SVM model on synthetic data on demand."""
        if self._is_trained:
            return True
        if not HAS_SKLEARN:
            return False
        try:
            self._train_classifier()
            self._is_trained = True
            logger.info("Math digit recognition classifier trained and ready.")
            return True
        except Exception as exc:
            self.error = f"Failed to train digit classifier: {exc}"
            logger.error(self.error)
            return False

    @staticmethod
    def _extract_features(mask: np.ndarray) -> np.ndarray:
        """Converts binary digit crop into normalized 64-dimensional feature vector."""
        m = (mask > 0).astype(np.uint8) * 255
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return np.zeros(64, dtype=np.float32)

        m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        h, w = m.shape
        side = int(max(h, w) * 8 / 6) + 2
        sq = np.zeros((side, side), np.uint8)
        oy, ox = (side - h) // 2, (side - w) // 2
        sq[oy:oy + h, ox:ox + w] = m

        small = cv2.resize(sq, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        if small.max() > 0:
            small = small / small.max() * 16.0
        return small.reshape(-1)

    def _generate_synthetic_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates synthetic digit samples under various fonts and affine transforms."""
        rng = np.random.default_rng(42)
        features_list: List[np.ndarray] = []
        labels_list: List[int] = []

        for digit in range(10):
            for font in self.FONTS:
                for thick in (5, 8, 11):
                    for _ in range(14):
                        scale = 2.6 if font != cv2.FONT_HERSHEY_PLAIN else 5.0
                        img = np.zeros((160, 160), np.uint8)
                        (tw, th), _ = cv2.getTextSize(str(digit), font, scale, thick)
                        cv2.putText(
                            img,
                            str(digit),
                            ((160 - tw) // 2, (160 + th) // 2),
                            font,
                            scale,
                            255,
                            thick,
                            cv2.LINE_AA,
                        )
                        ang = rng.uniform(-12, 12)
                        shear = rng.uniform(-0.2, 0.2)
                        sx = rng.uniform(0.8, 1.15)
                        M = cv2.getRotationMatrix2D((80, 80), ang, 1.0)
                        M[0, 1] += shear
                        M[0, 0] *= sx
                        warped = cv2.warpAffine(img, M, (160, 160))
                        features_list.append(self._extract_features(warped > 60))
                        labels_list.append(digit)

        return np.array(features_list), np.array(labels_list)

    def _train_classifier(self) -> None:
        """Trains an RBF-kernel SVM on synthetic digits and scikit-learn standard dataset."""
        synthetic_x, synthetic_y = self._generate_synthetic_samples()
        digits_dataset = load_digits()
        x_train = np.vstack([synthetic_x, digits_dataset.data.astype(np.float32)])
        y_train = np.concatenate([synthetic_y, digits_dataset.target])
        self.clf = SVC(kernel="rbf", gamma=0.002, C=10.0).fit(x_train, y_train)

    def _classify_digit(self, mask: np.ndarray) -> str:
        """Predicts digit character from image mask."""
        if self.clf is None:
            return "?"
        pred = self.clf.predict([self._extract_features(mask)])[0]
        return str(int(pred))

    @staticmethod
    def _cross_kind(mask: np.ndarray) -> Optional[str]:
        """Detects '+' or '*' from cross-like geometric symmetry."""
        h, w = mask.shape
        if h == 0 or w == 0 or not (0.55 < w / h < 1.8):
            return None

        g = cv2.resize(mask.astype(np.uint8) * 255, (32, 32), interpolation=cv2.INTER_AREA) > 40
        band = slice(11, 21)
        vcov = g[:, band].any(axis=1).mean()
        hcov = g[band, :].any(axis=0).mean()

        d1 = np.mean([g[max(0, i - 3):i + 4, max(0, i - 3):i + 4].any() for i in range(32)])
        d2 = np.mean([g[max(0, i - 3):i + 4, max(0, 31 - i - 3):31 - i + 4].any() for i in range(32)])

        corners = np.mean([g[:8, :8].mean(), g[:8, -8:].mean(), g[-8:, :8].mean(), g[-8:, -8:].mean()])
        mids = np.mean([g[:8, 12:20].mean(), g[-8:, 12:20].mean(), g[12:20, :8].mean(), g[12:20, -8:].mean()])
        cmax = max(g[:8, :8].mean(), g[:8, -8:].mean(), g[-8:, :8].mean(), g[-8:, -8:].mean())

        cs, rs = g.sum(axis=0), g.sum(axis=1)
        if cs.max() == 0 or rs.max() == 0:
            return None

        pc = int(round(np.nonzero(cs >= 0.9 * cs.max())[0].mean()))
        pr = int(round(np.nonzero(rs >= 0.9 * rs.max())[0].mean()))
        centred = abs(pc - 15.5) < 6 and abs(pr - 15.5) < 6

        quads = [
            g[:max(pr - 5, 1), :max(pc - 5, 1)],
            g[:max(pr - 5, 1), pc + 6:],
            g[pr + 6:, :max(pc - 5, 1)],
            g[pr + 6:, pc + 6:],
        ]
        quad_ink = max((q.mean() for q in quads if q.size), default=0.0)
        centred = centred and quad_ink < 0.1

        if min(vcov, hcov) > 0.8 and corners < 0.12 and cmax < 0.2 and centred:
            return "+"
        if min(d1, d2) > 0.8 and mids < 0.15:
            return "*"
        return None

    @staticmethod
    def _line_angle(mask: np.ndarray) -> Tuple[float, float]:
        """Calculates line orientation angle and linearity using PCA."""
        ys, xs = np.nonzero(mask)
        if len(xs) < 2:
            return 1.0, 0.0

        pts = np.column_stack([xs, ys]).astype(np.float32)
        pts -= pts.mean(axis=0)
        cov = np.cov(pts.T)
        vals, vecs = np.linalg.eigh(cov)
        lineness = np.sqrt(max(vals[0], 1e-6) / max(vals[1], 1e-6))
        vx, vy = vecs[:, 1]
        ang = abs(np.degrees(np.arctan2(vy, vx)))
        if ang > 90:
            ang = 180 - ang
        return float(lineness), float(ang)

    def _classify_symbol(self, comps: List[Dict[str, Any]], ref_h: int) -> str:
        """Determines symbol identity given grouped stroke components."""
        gx1 = min(c["bbox"][0] for c in comps)
        gy1 = min(c["bbox"][1] for c in comps)
        gx2 = max(c["bbox"][0] + c["bbox"][2] for c in comps)
        gy2 = max(c["bbox"][1] + c["bbox"][3] for c in comps)
        gw, gh = gx2 - gx1, gy2 - gy1
        gm = np.zeros((gh, gw), bool)

        for c in comps:
            x, y, w, h = c["bbox"]
            gm[y - gy1:y - gy1 + h, x - gx1:x - gx1 + w] |= c["mask"]

        flat = [c for c in comps if c["bbox"][2] > 2.0 * c["bbox"][3]]
        n = len(comps)

        if n == 2 and len(flat) == 2:
            return "="
        if n == 3 and len(flat) == 1:
            return "/"
        if n == 1:
            if gw > 2.2 * gh:
                return "-"
            if gw < 0.3 * ref_h and gh < 0.3 * ref_h:
                return "."

        cross = self._cross_kind(gm)
        if cross:
            return cross

        if n == 1:
            lineness, ang = self._line_angle(gm)
            if lineness < 0.2:
                if ang < 20:
                    return "-"
                if ang > 74:
                    return "1"
                return "/"

        if n > 3:
            return "?"
        return self._classify_digit(gm)

    def read_ink(self, ink_mask: np.ndarray) -> Optional[MathRecognitionResult]:
        """
        Segments connected components in the ink mask, groups multi-stroke symbols,
        and outputs parsed math expression.
        """
        if not self.ready or not self.ensure_trained():
            return None

        dilated = cv2.dilate(ink_mask, np.ones((7, 7), np.uint8))
        n, labels, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
        comps: List[Dict[str, Any]] = []

        for i in range(1, n):
            m = (labels == i) & (ink_mask > 0)
            if m.sum() < 25:
                continue
            ys, xs = np.nonzero(m)
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            comps.append({
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "mask": m[y1:y2, x1:x2],
            })

        if not comps:
            return None

        # Sort strokes from left to right
        comps.sort(key=lambda c: c["bbox"][0])

        # Merge horizontally overlapping strokes into single symbols
        groups: List[List[Dict[str, Any]]] = []
        for c in comps:
            x, y, w, h = c["bbox"]
            if groups:
                g = groups[-1]
                gx1 = min(k["bbox"][0] for k in g)
                gx2 = max(k["bbox"][0] + k["bbox"][2] for k in g)
                ov = min(gx2, x + w) - max(gx1, x)
                if ov > 0.35 * min(gx2 - gx1, w):
                    g.append(c)
                    continue
            groups.append([c])

        ref_h = max(
            max(k["bbox"][1] + k["bbox"][3] for k in g) - min(k["bbox"][1] for k in g)
            for g in groups
        )

        symbols: List[str] = []
        boxes: List[Tuple[int, int, int, int]] = []

        for g in groups:
            symbols.append(self._classify_symbol(g, ref_h))
            x1 = min(k["bbox"][0] for k in g)
            y1 = min(k["bbox"][1] for k in g)
            x2 = max(k["bbox"][0] + k["bbox"][2] for k in g)
            y2 = max(k["bbox"][1] + k["bbox"][3] for k in g)
            boxes.append((x1, y1, x2, y2))

        text = "".join(symbols)
        value, shown = self.evaluate(text)
        bx1 = min(b[0] for b in boxes)
        by1 = min(b[1] for b in boxes)
        bx2 = max(b[2] for b in boxes)
        by2 = max(b[3] for b in boxes)

        return MathRecognitionResult(
            symbols=symbols,
            boxes=boxes,
            text=text,
            value=value,
            shown=shown,
            bbox=(bx1, by1, bx2, by2),
        )

    @staticmethod
    def _safe_ast_eval(node: Any) -> float:
        """Recursively evaluates arithmetic AST safely with strict node whitelisting."""
        if isinstance(node, ast.Expression):
            return MathRecognizer._safe_ast_eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = MathRecognizer._safe_ast_eval(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = MathRecognizer._safe_ast_eval(node.left)
            right = MathRecognizer._safe_ast_eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError("division by zero")
            return left / right
        raise ValueError("Unsupported AST operation")

    def evaluate(self, text: str) -> Tuple[Optional[str], str]:
        """Safely parses and evaluates mathematical expression string."""
        expr = text.split("=")[0].rstrip("+-*/.")
        if not expr or "?" in expr or not any(ch.isdigit() for ch in expr):
            return None, expr
        if not any(op in expr[1:] for op in "+-*/"):
            return None, expr

        try:
            val = self._safe_ast_eval(ast.parse(expr, mode="eval"))
        except ZeroDivisionError:
            return "Div/0 Error", expr
        except Exception:
            return None, expr

        if abs(val - round(val)) < 1e-9:
            return str(int(round(val))), expr
        return f"{val:.4f}".rstrip("0").rstrip("."), expr
