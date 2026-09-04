## What Is IoU?

Intersection over Union (IoU), also called the Jaccard index, measures how much two regions overlap:

$$
\text{IoU} = \frac{|A \cap B|}{|A \cup B|} = \frac{\text{Area of Intersection}}{\text{Area of Union}}
$$

For bounding boxes:
- $A$ is the predicted bounding box
- $B$ is the ground truth bounding box
- The intersection is where they overlap
- The union is the total area covered by either box

---

## IoU Range and Interpretation

IoU ranges from 0 to 1:

**IoU = 0:** No overlap at all. The boxes are completely separate.
**IoU = 0.5:** Moderate overlap. Often used as a threshold for "correct" detection.
**IoU = 0.75:** Good overlap. Used for stricter evaluation (COCO AP75).
**IoU = 1.0:** Perfect overlap. Boxes are identical.

Common thresholds in object detection:
- IoU >= 0.5: "correct" detection in PASCAL VOC
- IoU >= 0.5, 0.55, ..., 0.95: averaged in COCO mAP

---

## Computing IoU for Axis-Aligned Boxes

Given two boxes defined by (x1, y1, x2, y2) where (x1, y1) is top-left and (x2, y2) is bottom-right:

**Step 1: Find intersection coordinates**
- inter_x1 = max(box1_x1, box2_x1)
- inter_y1 = max(box1_y1, box2_y1)
- inter_x2 = min(box1_x2, box2_x2)
- inter_y2 = min(box1_y2, box2_y2)

**Step 2: Compute intersection area**
- inter_width = max(0, inter_x2 - inter_x1)
- inter_height = max(0, inter_y2 - inter_y1)
- intersection = inter_width * inter_height

**Step 3: Compute union area**
- area1 = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
- area2 = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
- union = area1 + area2 - intersection

**Step 4: Compute IoU**
- IoU = intersection / union

---

## Numerical Example

Box A (predicted): x1=100, y1=100, x2=200, y2=200
Box B (ground truth): x1=120, y1=110, x2=220, y2=210

**Intersection:**
- inter_x1 = max(100, 120) = 120
- inter_y1 = max(100, 110) = 110
- inter_x2 = min(200, 220) = 200
- inter_y2 = min(200, 210) = 200
- inter_width = 200 - 120 = 80
- inter_height = 200 - 110 = 90
- intersection = 80 * 90 = 7200

**Areas:**
- area_A = (200-100) * (200-100) = 10000
- area_B = (220-120) * (210-110) = 10000
- union = 10000 + 10000 - 7200 = 12800

**IoU:** 7200 / 12800 = 0.5625

---

## IoU as a Loss Function

Using IoU directly as a loss:

$$
L_{\text{IoU}} = 1 - \text{IoU}
$$

This loss is 0 when boxes perfectly overlap and 1 when they do not overlap at all.

Advantages:
- Scale-invariant: same loss for large and small boxes with same relative overlap
- Directly optimizes the evaluation metric
- Considers all four box coordinates together

Disadvantage:
- Gradient is 0 when boxes do not overlap (IoU = 0)
- Cannot learn to move boxes toward each other if they start far apart

---

## The Non-Overlapping Problem

Consider two non-overlapping boxes:
- Box A: (0, 0, 10, 10)
- Box B: (100, 100, 110, 110)

IoU = 0, so IoU loss = 1.

Now move Box A slightly right:
- Box A: (1, 0, 11, 10)

Still no overlap, IoU = 0, loss = 1.

The gradient is zero! The model receives no signal about which direction to move. This is a critical limitation of vanilla IoU loss.

---

## GIoU: Generalized IoU

GIoU (Generalized Intersection over Union) fixes the non-overlapping problem:

$$
\text{GIoU} = \text{IoU} - \frac{|C - (A \cup B)|}{|C|}
$$

Where $C$ is the smallest enclosing box that contains both $A$ and $B$.

**Key insight:** even when boxes do not overlap, the enclosing box $C$ changes as boxes move. This provides gradient signal.

GIoU range: [-1, 1]
- GIoU = 1: perfect overlap
- GIoU = 0: boxes are adjacent
- GIoU < 0: boxes are far apart

Loss: $L_{\text{GIoU}} = 1 - \text{GIoU}$

---

## DIoU and CIoU

**DIoU (Distance IoU):** adds penalty for center distance

$$
\text{DIoU} = \text{IoU} - \frac{d^2}{c^2}
$$

Where:
- $d$ is the Euclidean distance between box centers
- $c$ is the diagonal of the enclosing box

This directly encourages boxes to have similar centers.

**CIoU (Complete IoU):** adds penalty for aspect ratio difference

$$
\text{CIoU} = \text{IoU} - \frac{d^2}{c^2} - \alpha v
$$

Where:
- $v$ measures aspect ratio consistency
- $\alpha$ is a trade-off parameter

CIoU considers overlap, center distance, and shape similarity.

---

## Comparison of IoU Variants

**Vanilla IoU:**
- Simple and intuitive
- Zero gradient for non-overlapping boxes
- Good for evaluation, problematic for training

**GIoU:**
- Handles non-overlapping boxes
- Can be slow to converge (tends to first enlarge, then shrink boxes)
- Good general-purpose choice

**DIoU:**
- Faster convergence than GIoU
- Directly optimizes center alignment
- Better for boxes that need to move far

**CIoU:**
- Most complete formulation
- Best overall performance in most benchmarks
- Slightly more complex to implement

---

## The Gradient of IoU

For IoU loss, the gradient with respect to box coordinates is non-trivial because it involves min/max operations.

For predicted box coordinates (x1, y1, x2, y2):
- Gradient flows through the intersection computation
- Only active when the coordinate is on the "boundary" of the intersection
- This is why IoU loss can have sparse gradients

Modern deep learning frameworks handle this automatically through autograd.

---

## Where IoU Loss Is Used

- **Object detection**: YOLO, Faster R-CNN, RetinaNet all use IoU-based losses
- **Instance segmentation**: mask IoU for evaluating predicted masks
- **Tracking**: measuring how well a tracker follows an object
- **Image registration**: aligning images or regions
- **Any task involving bounding box regression**

Best practices:
- Use GIoU or CIoU for training (better gradients)
- Use IoU for evaluation (standard metric)
- Combine with classification loss for detection (total loss = classification + box regression)