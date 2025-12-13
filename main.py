import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from sklearn.preprocessing import normalize
import joblib
import os
import copy
import chess
import chess.engine
import time

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DINOv2 Transforms
dino_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 2. VISION PIPELINE (Geometry & Warping)
# ==========================================

def resize_image(img, max_width=1024):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        return cv2.resize(img, (max_width, new_h))
    return img

def pre_processing_pipeline(image, threshold1=10, threshold2=200):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, threshold1, threshold2)
    return edges

def detect_lines(edges, hough_threshold=100, min_line_length=50, max_line_gap=5):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    return lines if lines is not None else []

def robust_chessboard_lines(lines):
    """Separates lines into Horizontal and Vertical clusters based on angle."""
    v_lines = []
    h_lines = []
    if lines is None: return [], []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 45 or angle > 135:
            h_lines.append(line[0])
        else:
            v_lines.append(line[0])
    return v_lines, h_lines

def compute_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0: return [0, 0]
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [px, py]

def get_4_points(h_lines, v_lines):
    # Sort H by Y (Top to Bottom), V by X (Left to Right)
    h_lines = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
    v_lines = sorted(v_lines, key=lambda l: (l[0] + l[2]) / 2)

    top, bottom = h_lines[0], h_lines[-1]
    left, right = v_lines[0], v_lines[-1]

    p_tl = compute_intersection(top, left)
    p_tr = compute_intersection(top, right)
    p_br = compute_intersection(bottom, right)
    p_bl = compute_intersection(bottom, left)
    
    return np.float32([p_tl, p_tr, p_br, p_bl])

def warp_to_chessboard_plane(image, inter_points, output_size=(1024, 1024), 
                             pad_top=0.1, pad_bottom=0.05, pad_sides=0.05):
    w, h = output_size
    px_top = h * pad_top
    px_bottom = h * pad_bottom
    px_left = w * pad_sides
    px_right = w * pad_sides 

    dst_points = np.float32([
        [px_left,      px_top],                 
        [w - px_right, px_top],                 
        [w - px_right, h - px_bottom],          
        [px_left,      h - px_bottom]           
    ])
    M = cv2.getPerspectiveTransform(inter_points, dst_points)
    warped = cv2.warpPerspective(image, M, output_size)
    return warped, M

# --- Grid Helpers ---
def split_lines_hv(lines, angle_tolerance_deg=5):
    h_lines, v_lines = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = np.abs(angle)
        if angle > 90: angle = 180 - angle
        
        if angle < angle_tolerance_deg: h_lines.append(line[0])
        elif abs(angle - 90) < angle_tolerance_deg: v_lines.append(line[0])
    return h_lines, v_lines

def collapse_and_average_lines(lines, dimension_index, grouping_threshold=15):
    if not lines: return []
    sorted_lines = sorted(lines, key=lambda l: (l[dimension_index] + l[dimension_index+2])/2)
    merged = []
    current_group = [sorted_lines[0]]
    
    for line in sorted_lines[1:]:
        curr_pos = (line[dimension_index] + line[dimension_index+2])/2
        prev_pos = (current_group[-1][dimension_index] + current_group[-1][dimension_index+2])/2
        if abs(curr_pos - prev_pos) < grouping_threshold:
            current_group.append(line)
        else:
            merged.append(np.mean(current_group, axis=0).astype(int))
            current_group = [line]
    merged.append(np.mean(current_group, axis=0).astype(int))
    return merged

def generate_final_grid_points(v_lines, h_lines):
    # Sort
    v_lines = sorted(v_lines, key=lambda l: (l[0]+l[2])/2)
    h_lines = sorted(h_lines, key=lambda l: (l[1]+l[3])/2)
    
    # We expect 9 lines for 8x8. If fewer, we could interpolate, but assuming clean data for now.
    points = np.zeros((len(h_lines), len(v_lines), 2), dtype=np.float32)
    for i, h_line in enumerate(h_lines):
        for j, v_line in enumerate(v_lines):
            pt = compute_intersection(h_line, v_line)
            points[i, j] = pt
    return points

def extract_tiles_and_pieces_dynamic(warped_image, grid_points, corner_pts,
                                     left=10, right=10, bottom=5, 
                                     base_top=30, scale_factor=3.5):
    tiles, pieces = [], []
    board_h, board_w = warped_image.shape[:2]
    ideal_tile_h = board_h / 8

    # Distortion Heuristic
    orig_w = np.linalg.norm(corner_pts[1] - corner_pts[0])
    orig_h = np.linalg.norm(corner_pts[3] - corner_pts[0])
    distortion_ratio = (orig_h / orig_w) * 3  

    # Limit to 8x8
    rows = min(8, grid_points.shape[0]-1)
    cols = min(8, grid_points.shape[1]-1)

    for i in range(rows):      
        for j in range(cols):  
            x_min = int(min(grid_points[i,j][0], grid_points[i+1,j][0]))
            x_max = int(max(grid_points[i,j+1][0], grid_points[i+1,j+1][0]))
            y_min = int(min(grid_points[i,j][1], grid_points[i,j+1][1]))
            y_max = int(max(grid_points[i+1,j][1], grid_points[i+1,j+1][1]))

            # 1. TILES (Strict Crop)
            pad_x, pad_y = int((x_max-x_min)*0.1), int((y_max-y_min)*0.1)
            t_y1, t_y2 = max(0, y_min+pad_y), min(board_h, y_max-pad_y)
            t_x1, t_x2 = max(0, x_min+pad_x), min(board_w, x_max-pad_x)
            tile_img = warped_image[t_y1:t_y2, t_x1:t_x2].copy()
            tiles.append(tile_img)

            # 2. PIECES (Dynamic Crop)
            row_factor = (7 - i) / 7.0 
            extra_headroom = int(base_top + (row_factor * abs(1 - distortion_ratio) * scale_factor * ideal_tile_h))
            y_min_dyn = max(0, y_min - extra_headroom)
            y_max_dyn = min(board_h, y_max + bottom)
            x_min_dyn = max(0, x_min - left)
            x_max_dyn = min(board_w, x_max + right)
            piece_img = warped_image[y_min_dyn:y_max_dyn, x_min_dyn:x_max_dyn].copy()
            pieces.append(piece_img)

    return tiles, pieces

# ==========================================
# 3. AI / INFERENCE
# ==========================================

def load_dino():
    print("⏳ Loading DINOv2 model...")
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
    dino.eval()
    return dino

def extract_features(image_list, dino_model):
    batch_tensors = []
    from PIL import Image
    for img in image_list:
        if isinstance(img, np.ndarray):
            if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        # Resize tiny crops
        if img.size[0] < 5: img = img.resize((224,224))
        batch_tensors.append(dino_transform(img))
        
    if not batch_tensors: return np.array([])
    batch_tensor = torch.stack(batch_tensors).to(DEVICE)
    with torch.no_grad():
        features = dino_model(batch_tensor).cpu().numpy()
    return normalize(features, norm='l2')

def apply_global_constraints(board_probs, train_classes, svm_classes):
    corrected_probs = copy.deepcopy(board_probs)
    # Piece limits
    PIECE_LIMITS = {
        'white_king': 1, 'black_king': 1,
        'white_queen': 1, 'black_queen': 1,
        'white_rook': 2, 'black_rook': 2,
        'white_bishop': 2, 'black_bishop': 2,
        'white_knight': 2, 'black_knight': 2,
        'white_pawn': 8, 'black_pawn': 8
    }
    for piece_name, limit in PIECE_LIMITS.items():
        try:
            name_idx = train_classes.index(piece_name)
            col_idx = np.where(svm_classes == name_idx)[0][0]
        except: continue

        candidates = []
        for i in range(64):
            probs = corrected_probs[i]
            if probs is not None:
                candidates.append((i, probs[col_idx]))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        for i, (square_idx, prob) in enumerate(candidates):
            if i >= limit:
                corrected_probs[square_idx][col_idx] = 0
                total = np.sum(corrected_probs[square_idx])
                if total > 0: corrected_probs[square_idx] /= total
    return corrected_probs

def get_final_board_state(tiles, pieces, clf_occ, clf_piece, classes, dino, ghost_threshold=0.45):
    # 1. Features
    occ_feats = extract_features(tiles, dino)
    piece_feats = extract_features(pieces, dino)
    
    occupancy_states = []
    raw_board_probs = [None] * 64
    
    # 2. Occupancy Predictions
    for i in range(len(tiles)):
        feat = occ_feats[i].reshape(1, -1)
        prob = clf_occ.predict_proba(feat)[0]
        is_occupied = prob[1] > 0.5
        occupancy_states.append(is_occupied)
        if is_occupied:
            feat_p = piece_feats[i].reshape(1, -1)
            raw_board_probs[i] = clf_piece.predict_proba(feat_p)[0]
            
    # 3. Apply Logic (One King, etc)
    final_probs = apply_global_constraints(raw_board_probs, classes, clf_piece.classes_)
    
    # 4. Build List
    board_state = []
    for i in range(len(tiles)):
        if not occupancy_states[i]:
            board_state.append("empty")
        else:
            probs = final_probs[i]
            top_idx = np.argmax(probs)
            top_prob = probs[top_idx]
            
            # Map back to class name
            if hasattr(clf_piece, 'classes_'):
                 # Check if classes_ are ints (indices) or strings (names)
                 if isinstance(clf_piece.classes_[0], (int, np.integer)):
                     class_name = classes[clf_piece.classes_[top_idx]]
                 else:
                     class_name = clf_piece.classes_[top_idx]
            else:
                 class_name = classes[top_idx]

            if top_prob < ghost_threshold:
                board_state.append("empty")
            else:
                board_state.append(class_name)
    return board_state

# ==========================================
# 4. RESULTS & STOCKFISH
# ==========================================

def board_to_fen(board_list):
    fen_map = {
        "empty": None,
        "wk": "K", "wq": "Q", "wr": "R", "wb": "B", "wn": "N", "wp": "P",
        "bk": "k", "bq": "q", "br": "r", "bb": "b", "bn": "n", "bp": "p"
    }
    fen_rows = []
    for row in range(8):
        empty_count = 0
        row_str = ""
        for col in range(8):
            label = board_list[row * 8 + col]
            key = label.lower().replace("white_", "w").replace("black_", "b")
            if "knight" in key: key = key.replace("knight", "n")
            
            char = fen_map.get(key, "?")
            if char is None: empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += char
        if empty_count > 0: row_str += str(empty_count)
        fen_rows.append(row_str)
    return "/".join(fen_rows) + " w - - 0 1"

def get_best_move(fen_string, stockfish_path, time_limit=2.0):
    try:
        board = chess.Board(fen_string)
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))
            score = info["score"].white() if board.turn == chess.WHITE else info["score"].black()
            val = f"Mate {score.mate()}" if score.is_mate() else f"{score.score()/100:+.2f}"
            print(f"   Evaluation: {val}")
            return board.san(result.move)
    except Exception as e:
        print(f"❌ Stockfish Error: {e}")
        return None

def visualize_fen_debug(fen):
    # Simple matplotlib viz
    piece_map = {'K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙', 
                 'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟'}
    board_grid = np.zeros((8, 8))
    for r in range(8):
        for c in range(8):
            board_grid[r, c] = 1 if (r+c)%2==0 else 0.7

    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(board_grid, cmap='gray', vmin=0, vmax=1)
    
    rows = fen.split()[0].split('/')
    for r, row_str in enumerate(rows):
        c = 0
        for char in row_str:
            if char.isdigit(): c += int(char)
            else:
                color = 'cyan' if char.isupper() else 'black'
                ax.text(c, r, piece_map.get(char, char), fontsize=24, 
                        ha='center', va='center', color=color, fontweight='bold')
                c += 1
    plt.title(f"FEN: {fen}", fontsize=9)
    plt.show()

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def solve_pipeline(image_path, stockfish_path, debug=False):
    print(f"\n🚀 STARTING PIPELINE: {image_path}")
    
    # 1. LOAD SAVED MODELS
    if not (os.path.exists('model_occupancy.pkl') and os.path.exists('model_piece.pkl')):
        print("❌ Error: Models not found. Please save them from your notebook first!")
        return

    print("🧠 Loading AI Models...")
    clf_occ = joblib.load('model_occupancy.pkl')
    clf_piece = joblib.load('model_piece.pkl')
    classes = joblib.load('model_classes.pkl')
    dino = load_dino()

    # 2. VISION
    img = cv2.imread(image_path)
    if img is None: 
        print("❌ Error: Image not found.")
        return
    img = resize_image(img)
    
    if debug:
        cv2.imshow("Original", img)
        cv2.waitKey(500)

    # Detect Lines
    edges = pre_processing_pipeline(img)
    lines = detect_lines(edges)
    v_lines, h_lines = robust_chessboard_lines(lines)
    
    if debug:
        dbg = img.copy()
        for l in v_lines: cv2.line(dbg, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
        for l in h_lines: cv2.line(dbg, (l[0],l[1]), (l[2],l[3]), (0,0,255), 2)
        cv2.imshow("Detected Lines", dbg)
        cv2.waitKey(500)

    # Warp
    try:
        corner_pts = get_4_points(h_lines, v_lines)
        warped, M = warp_to_chessboard_plane(img, corner_pts)
    except Exception as e:
        print(f"❌ Vision Failed: {e}")
        return

    if debug:
        cv2.imshow("Warped Board", warped)
        cv2.waitKey(500)

    # Grid Detection (Notebook Logic)
    w_edges = pre_processing_pipeline(warped, 10, 150)
    w_lines = detect_lines(w_edges, 100, 75, 3)
    wh_lines, wv_lines = split_lines_hv(w_lines)
    
    # Refine grid
    c_wh = collapse_and_average_lines(wh_lines, 1, 15)
    c_wv = collapse_and_average_lines(wv_lines, 0, 15)
    
    # Make points
    grid_points = generate_final_grid_points(c_wv, c_wh)
    
    # Extract Crops
    tiles, pieces = extract_tiles_and_pieces_dynamic(warped, grid_points, corner_pts)
    print(f"✅ Extracted {len(tiles)} squares.")

    # 3. CLASSIFICATION
    print("🧠 Analyzing pieces...")
    final_classes = get_final_board_state(tiles, pieces, clf_occ, clf_piece, classes, dino, 0.45)
    
    # 4. RESULTS
    fen = board_to_fen(final_classes)
    print(f"\n📋 FEN: {fen}")
    
    if debug:
        visualize_fen_debug(fen)
        if cv2.getWindowProperty("Warped Board", 0) >= 0:
            cv2.destroyAllWindows()

    if stockfish_path:
        print("\n🤖 Stockfish Thinking...")
        best_move = get_best_move(fen, stockfish_path)
        print(f"\n🏆 BEST MOVE: {best_move}")

if __name__ == "__main__":
    # --- USER SETTINGS ---
    MY_IMAGE = "boards/board1.jpg"
    MY_STOCKFISH = r"C:\stockfish\stockfish-windows-x86-64.exe" 
    
    # Set this to True to see all the steps!
    DEBUG_MODE = True  
    
    solve_pipeline(MY_IMAGE, MY_STOCKFISH, debug=DEBUG_MODE)