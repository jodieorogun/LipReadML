import os
import cv2
import dlib
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
 
 # setting up dlibs face detector
PREDICTOR_PATH = "data/shape_predictor_68_face_landmarks.dat" 
DLIB_FACE = dlib.get_frontal_face_detector()
DLIB_SHAPE = dlib.shape_predictor(PREDICTOR_PATH)
MOUTH_IDX = list(range(48, 68))  # mouth landmarks in model

# cropping with boundary checks 
def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w, int(x2)); y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


class MouthCropperDlib: # dlib-based mouth cropper with smoothing
    def __init__(self, pad=1.5, smooth=0.6, det_width=320, detect_every=10):
        self.pad = pad
        self.smooth = smooth
        self.prev = None
        self.det_width = det_width
        self.detect_every = detect_every
        self._frame_counter = 0

    def _ema(self, new_box): # smoothing 
        if self.prev is None:
            self.prev = new_box
            return new_box
        a = self.smooth
        px1, py1, px2, py2 = self.prev
        x1, y1, x2, y2 = new_box
        sm = (int(a*px1 + (1-a)*x1),
              int(a*py1 + (1-a)*y1),
              int(a*px2 + (1-a)*x2),
              int(a*py2 + (1-a)*y2))
        self.prev = sm
        return sm

    def _detect_face_rect(self, gray): # returns dlib.rectangle for detected face
        h, w = gray.shape[:2]
        scale = 1.0
        if w > self.det_width:
            scale = self.det_width / float(w)
            small = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        else:
            small = gray
        rects = DLIB_FACE(small, 0)
        if not rects:
            return None
        r = max(rects, key=lambda rr: rr.width()*rr.height())
        inv = 1.0/scale
        return dlib.rectangle(int(r.left()*inv), int(r.top()*inv),
                              int(r.right()*inv), int(r.bottom()*inv))

    def detect_bbox(self, frame_bgr): # returns (x1,y1,x2,y2) for coords around mouth
        self._frame_counter += 1
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        need_detect = (self.prev is None) or (self._frame_counter % self.detect_every == 0)
        if need_detect:
            rect = self._detect_face_rect(gray)
            if rect is None:
                return self.prev
            shape = DLIB_SHAPE(gray, rect)
            pts = np.array([(shape.part(i).x, shape.part(i).y) for i in MOUTH_IDX], dtype=np.float32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
            side = max(x2-x1, y2-y1) * self.pad
            box = (int(cx - side/2), int(cy - side/2), int(cx + side/2), int(cy + side/2))
            return self._ema(box)
        else:
            return self.prev

mouth_cropper = MouthCropperDlib() 

def parseAlignFile(pathToAlign, word_index=0):  # getting the target word from .align file
    words = []
    with open(pathToAlign, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            token = parts[2].lower()
            if token not in ("sil", "sp"):
                words.append(token)
    return words[word_index] if len(words) > word_index else None

def hog_feat(gray): #HOG feature extraction
    return hog(gray, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), block_norm="L2-Hys",
               transform_sqrt=True, feature_vector=True)

def videoToFeature(path, max_frames=64, debug=False): # video to feature vector
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or total <= 0:
        cap.release()
        return None

    target_indices = np.linspace(0, total-1, num=min(total, max_frames*3)).astype(int)
    idx_set = set(target_indices.tolist())

    framesHists = [] # collecting HOG features
    i = -1
    while True:
        grabbed, frame = cap.read() # read frame
        if not grabbed:
            break
        i += 1
        if i not in idx_set:
            continue

        bbox = mouth_cropper.detect_bbox(frame) # detect mouth
        mouth = None
        if bbox:
            x1, y1, x2, y2 = bbox
            mouth = safe_crop(frame, x1, y1, x2, y2) # crop mouth
        if mouth is None: # fallback to center crop
            h, w = frame.shape[:2]
            side = int(0.5 * min(h, w))
            cx, cy = w//2, int(h*0.7)
            mouth = safe_crop(frame, cx-side//2, cy-side//2, cx+side//2, cy+side//2)
            if mouth is None:
                continue

        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY) # to grayscale       
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray) 

        feat = hog_feat(gray) # extract HOG feature
        framesHists.append(feat)   

        if debug:
            dbg = frame.copy()
            if bbox:
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.imshow("dbg_frame", dbg)
            cv2.imshow("dbg_mouth", gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(framesHists) >= max_frames:
            break

    cap.release()
    if debug:
        cv2.destroyAllWindows()
    if not framesHists:
        return None

    F = np.stack(framesHists, axis=0) 
    feat_mean, feat_std = F.mean(axis=0), F.std(axis=0)  
    return np.concatenate([feat_mean, feat_std]).astype(np.float32) # final feature vector


def buildDataset(gridRoot, word_index=0): # building dataset from GRID
    X, labels = [], []
    root = Path(gridRoot)
    speakers = sorted([p.name for p in root.glob("s*") if p.is_dir()])

    for sid in speakers:
        videos = root / sid / "videos"
        aligns = root / sid / "align"
        if not videos.exists() or not aligns.exists():
            continue

        for mpg in videos.glob("*.mpg"):
            cap = cv2.VideoCapture(str(mpg))
            if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 0:
                cap.release()
                continue
            cap.release()

            base = mpg.stem
            alignPath = aligns / f"{base}.align"
            if not alignPath.exists():
                continue

            label = parseAlignFile(alignPath, word_index=word_index)
            if not label:
                continue

            feat = videoToFeature(mpg, max_frames=96, debug=True) 
            if feat is None:
                continue
            # adding features and labels 
            X.append(feat) 
            labels.append(label)

    X = np.array(X, dtype=np.float32)
    labels = np.array(labels, dtype=object)
    return X, labels

def train(X, labels, k=3): # training
    print("Class counts:", Counter(labels)) 
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=32, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean") 
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    return knn, le # returning model and label encoder

if __name__ == "__main__":
    GRID_ROOT = "GRID"
    X, labels = buildDataset(GRID_ROOT, word_index=0)  # change word_index here (0=first, 1=second)
    print(f"Built dataset: X={X.shape} labels={len(labels)} unique={len(set(labels))}")
    knn, le = train(X, labels, k=3)
