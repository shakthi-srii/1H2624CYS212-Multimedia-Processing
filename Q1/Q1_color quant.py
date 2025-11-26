# color_quant.py
from PIL import Image
import numpy as np
from collections import defaultdict, deque

# ------------------------------
# 1) Greyscale via Desaturation
# ------------------------------
def desaturation_quantize(img: Image.Image, gray_levels: int) -> Image.Image:
    """Convert to grayscale via luminance and quantize to `gray_levels` levels."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    # luminance
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    # normalize 0..255 -> 0..(gray_levels-1)
    step = 255.0 / (gray_levels - 1)
    q = np.round(lum / step) * step
    q = q.clip(0, 255).astype(np.uint8)
    out = np.stack([q, q, q], axis=-1)
    return Image.fromarray(out, mode="RGB")

# ------------------------------
# 2) Median Cut Quantization
# ------------------------------
class ColorBox:
    def __init__(self, indices, pixels):
        self.indices = indices  # indices into pixels list
        self.pixels = pixels

    def get_range(self):
        pts = np.array([self.pixels[i] for i in self.indices], dtype=np.int32)
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        return maxs - mins

    def longest_channel(self):
        return int(np.argmax(self.get_range()))

    def average_color(self):
        pts = np.array([self.pixels[i] for i in self.indices], dtype=np.int64)
        avg = np.round(pts.mean(axis=0)).astype(np.uint8)
        return tuple(int(x) for x in avg.tolist())

def median_cut_quantize(img: Image.Image, k_colors: int) -> Image.Image:
    """Median cut quantization to k_colors."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    pixels = list(img.getdata())
    n = len(pixels)
    # start with one box
    boxes = [ColorBox(list(range(n)), pixels)]
    # split until we have at least k_colors boxes
    while len(boxes) < k_colors:
        # pick box with largest color range (by max span)
        boxes.sort(key=lambda b: max(b.get_range()), reverse=True)
        box = boxes.pop(0)
        if len(box.indices) <= 1:
            boxes.append(box)
            break
        ch = box.longest_channel()
        # sort indices by that channel and split at median
        box.indices.sort(key=lambda i: pixels[i][ch])
        mid = len(box.indices) // 2
        b1 = ColorBox(box.indices[:mid], pixels)
        b2 = ColorBox(box.indices[mid:], pixels)
        boxes.extend([b1, b2])
    # build mapping index -> color
    index_to_color = {}
    for box in boxes:
        col = box.average_color()
        for idx in box.indices:
            index_to_color[idx] = col
    # create output image
    out_pixels = [index_to_color[i] for i in range(n)]
    out = Image.new("RGB", img.size)
    out.putdata(out_pixels)
    return out

# ------------------------------
# 3) Octree Quantization (variant 2)
# ------------------------------
class OctreeNode:
    def __init__(self, level, parent):
        self.children = [None] * 8
        self.is_leaf = (level == 8)  # leaves at depth 8 by default
        self.pixel_count = 0
        self.red_sum = 0
        self.green_sum = 0
        self.blue_sum = 0
        self.parent = parent
        self.level = level

    def accumulate(self, color):
        r, g, b = color
        self.pixel_count += 1
        self.red_sum += r
        self.green_sum += g
        self.blue_sum += b

    def get_average(self):
        if self.pixel_count == 0:
            return (0, 0, 0)
        return (self.red_sum // self.pixel_count,
                self.green_sum // self.pixel_count,
                self.blue_sum // self.pixel_count)

class OctreeQuantizer:
    def __init__(self, max_colors=256):
        self.root = OctreeNode(0, None)
        self.leaf_count = 0
        self.max_colors = max_colors
        # reducible nodes per level (1..7)
        self.reducible = defaultdict(list)

    def _get_color_index(self, color, level):
        r, g, b = color
        shift = 8 - level
        r_bit = (r >> shift) & 1
        g_bit = (g >> shift) & 1
        b_bit = (b >> shift) & 1
        return (r_bit << 2) | (g_bit << 1) | b_bit

    def add_color(self, color):
        node = self.root
        for level in range(1, 9):  # levels 1..8
            idx = self._get_color_index(color, level)
            if node.children[idx] is None:
                node.children[idx] = OctreeNode(level, node)
                # if not leaf, it's reducible (levels 1..7)
                if level < 8:
                    self.reducible[level].append(node.children[idx])
                else:
                    self.leaf_count += 1
            node = node.children[idx]
            if level == 8:
                node.is_leaf = True
                node.accumulate(color)
        # ensure leaf_count accounted if nodes earlier turned into leaves:
        if node.is_leaf:
            # accumulation done; leaf_count was incremented upon creation
            pass

    def _reduce_once(self):
        # find deepest level with reducible nodes
        for lvl in range(7, 0, -1):
            if self.reducible.get(lvl):
                node = self.reducible[lvl].pop()
                # merge children into this node
                red_sum = green_sum = blue_sum = count = 0
                for i in range(8):
                    child = node.children[i]
                    if child:
                        red_sum += child.red_sum
                        green_sum += child.green_sum
                        blue_sum += child.blue_sum
                        count += child.pixel_count
                        # if child was non-leaf and present in reducible lists, remove it
                        if child.level < 8 and child in self.reducible.get(child.level, []):
                            try:
                                self.reducible[child.level].remove(child)
                            except ValueError:
                                pass
                        node.children[i] = None
                        if child.is_leaf:
                            self.leaf_count -= 1
                node.is_leaf = True
                node.red_sum = red_sum
                node.green_sum = green_sum
                node.blue_sum = blue_sum
                node.pixel_count = count
                self.leaf_count += 1
                return True
        return False

    def build_palette(self):
        palette = []
        # traverse tree and collect leaves
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is None:
                continue
            if node.is_leaf and node.pixel_count > 0:
                palette.append(node.get_average())
            else:
                for ch in node.children:
                    if ch is not None:
                        stack.append(ch)
        return palette

    def quantize(self, img: Image.Image) -> Image.Image:
        if img.mode != "RGB":
            img = img.convert("RGB")
        pixels = list(img.getdata())
        # build tree
        for color in pixels:
            self.add_color(color)
            # if leaves exceed allowed palette, reduce
            while self.leaf_count > self.max_colors:
                reduced = self._reduce_once()
                if not reduced:
                    break
        # ensure palette built
        # create a mapping color->palette by walking tree for each pixel
        def find_leaf_color(color):
            node = self.root
            for level in range(1, 9):
                idx = self._get_color_index(color, level)
                if node.children[idx] is None:
                    # if missing child, return current node if leaf, else fallback to averages
                    break
                node = node.children[idx]
                if node.is_leaf:
                    break
            return node.get_average()
        out_pixels = [find_leaf_color(c) for c in pixels]
        out = Image.new("RGB", img.size)
        out.putdata(out_pixels)
        return out

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    import sys
    img_path = "example.jpg"  # change to your file
    try:
        src = Image.open(img_path)
    except FileNotFoundError:
        # generate a test image if example.jpg not present
        print("example.jpg not found — generating test gradient image.")
        w, h = 320, 240
        src = Image.new("RGB", (w, h))
        for y in range(h):
            for x in range(w):
                src.putpixel((x, y), (int(255 * x / w), int(255 * y / h), int(128 + 127 * (x / w))))
        src.save("example_generated.jpg")
        img_path = "example_generated.jpg"
        src = Image.open(img_path)

    # parameters
    gray_levels = 8
    k = 16  # palette size for color quantizers

    # 1) desaturation
    out_desat = desaturation_quantize(src, gray_levels)
    out_desat.save("out_desaturation.png")
    print("Saved out_desaturation.png")

    # 2) median cut
    out_med = median_cut_quantize(src, k)
    out_med.save("out_mediancut.png")
    print("Saved out_mediancut.png")

    # 3) octree (variant 2)
    oct_q = OctreeQuantizer(max_colors=k)
    out_oct = oct_q.quantize(src)
    out_oct.save("out_octree.png")
    print("Saved out_octree.png")

    print("Done. Outputs: out_desaturation.png, out_mediancut.png, out_octree.png")
