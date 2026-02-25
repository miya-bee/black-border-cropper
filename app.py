import os
import sys
import re
import queue
import threading
import traceback
import tempfile
from datetime import datetime
from pathlib import Path

# =========================
# アプリ共通定数
# =========================
APP_VERSION = "v01_20"
APP_TITLE = f"Black Border Cropper {APP_VERSION} (PNG/JPG/JPEG/PDF + Folder + Manual Crop)"

# UI/非同期の調整値（全体）
BATCH_POLL_MS = 100
RESIZE_DEBOUNCE_MS = 80
DRAG_STATUS_THROTTLE_MS = 80  # 将来の間引き用（現状はドラッグ中ステータス抑制）

# UI表示・出力まわりの調整値
MIN_BBOX_SIZE = 5
BATCH_ERROR_PREVIEW_LIMIT = 10
SINGLE_OUTPUT_PREVIEW_LIMIT = 5
OUTPUT_NAME_DUPLICATE_MAX = 10000
JPEG_EXPORT_QUALITY = 95

# 出力処理で「ユーザーに返す失敗」として扱う例外（設計上の不具合はワーカー境界まで送る）
EXPORT_HANDLED_EXCEPTIONS = (OSError, ValueError, RuntimeError)

# UI/スレッド状態確認で握りつぶしてよい例外（終了処理・状態取得系）
THREAD_STATE_HANDLED_EXCEPTIONS = (AttributeError, RuntimeError)
TK_UI_HANDLED_EXCEPTIONS = (RuntimeError, ValueError)
# UIイベント境界で「ユーザー通知して継続」にしてよい運用時例外（設計不備系は極力含めない）
UI_ACTION_HANDLED_EXCEPTIONS = (OSError, RuntimeError, ValueError)
# 値変換/属性参照のフォールバックで握りつぶしてよい例外
COERCE_HANDLED_EXCEPTIONS = (TypeError, ValueError, AttributeError)
# Path/ファイル名の正規化で握りつぶしてよい例外（入力不正/OS依存）
PATH_NORMALIZE_HANDLED_EXCEPTIONS = (OSError, RuntimeError, TypeError, ValueError)
# ログ初期化/書き込み・スレッド起動など基盤処理のフォールバックで握りつぶしてよい例外
INFRA_HANDLED_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError, AttributeError)
# DnD文字列の分解（Tk境界）で握りつぶしてよい例外
DND_PARSE_HANDLED_EXCEPTIONS = (AttributeError, RuntimeError, ValueError, TypeError)

# 終了時クリーンアップ（ワーカーjoin）
CLOSE_WORKER_JOIN_TIMEOUT_MS = 200  # 1回のjoin待機スライス
CLOSE_WORKER_JOIN_TIMEOUT_SEC = CLOSE_WORKER_JOIN_TIMEOUT_MS / 1000.0
CLOSE_WORKER_JOIN_TOTAL_DEFAULT_MS = 1200
CLOSE_WORKER_JOIN_TOTAL_BATCH_MS = 1800
CLOSE_WORKER_JOIN_TOTAL_SINGLE_EXPORT_MS = 2200
CLOSE_WORKER_JOIN_TOTAL_SINGLE_EXPORTING_MS = 5000

# UIイベント（batch/single共通キュー）の種別
UI_EVENT_STATUS = "status"
UI_EVENT_DONE = "done"
UI_EVENT_FATAL = "fatal"
UI_EVENT_SINGLE_STATUS = "single_status"
UI_EVENT_SINGLE_DONE = "single_done"
UI_EVENT_SINGLE_FATAL = "single_fatal"

# UIイベント payload 契約（不足時は受信側で既定値補完するが、開発時の検知用に定義）
# - status / single_status: {text}
# - done: {total, ok_count, ng_count, skipped, canceled, errors}
# - single_done: {action, src, result}
# - fatal / single_fatal: {user_message, detail? , traceback?}
UI_EVENT_REQUIRED_KEYS = {
    UI_EVENT_STATUS: ("text",),
    UI_EVENT_SINGLE_STATUS: ("text",),
    UI_EVENT_DONE: ("total", "ok_count", "ng_count", "skipped", "canceled", "errors"),
    UI_EVENT_SINGLE_DONE: ("action", "src", "result"),
    UI_EVENT_FATAL: ("user_message",),
    UI_EVENT_SINGLE_FATAL: ("user_message",),
}

# 単一出力ワーカーの進捗フェーズ（終了時メッセージ/ログ改善用）
SINGLE_EXPORT_PHASE_IDLE = "idle"
SINGLE_EXPORT_PHASE_PREPARE = "prepare"
SINGLE_EXPORT_PHASE_EXPORTING = "exporting"
SINGLE_EXPORT_PHASE_POSTING = "posting_ui_event"

# 新ロジック（背景色ベース検出）の調整用定数
# ---------------------------------------------------------------------
# 方針:
# - ユーザー設定（しきい値 / 背景モード）から間接的に効く値 と
#   内部ヒューリスティック（実装都合の安定化パラメータ）をコメントで区別する。
# - UIに直接出す候補は絞り、以下は基本的に内部専用として扱う。
#   （将来チューニング時に「どの症状を抑える値か」を追いやすくする）
# ---------------------------------------------------------------------

# [A] 背景色推定（外周帯サンプリング）: 内部専用ヒューリスティック
EDGE_BG_BAND_RATIO = 0.03          # 外周帯の厚み（画像短辺に対する比率）
EDGE_BG_BAND_MAX_DIVISOR = 5       # 外周帯の厚みの上限制御（短辺/5まで）
EDGE_BG_QUANT_STEP = 12            # 背景色の量子化幅（大きいほどノイズに強いが色差に鈍い）
EDGE_BG_DOMINANT_MIN_SAMPLES = 32  # 最頻色binを採用する最低サンプル数
EDGE_BG_DOMINANT_MIN_RATIO = 0.02  # 最頻色binを採用する最低比率（弱い時は中央値へフォールバック）
EDGE_BG_DEV_P90_PERCENTILE = 90.0  # 背景揺らぎの下限側目安（通常ノイズ）
EDGE_BG_DEV_P98_PERCENTILE = 98.0  # 背景揺らぎの上限側目安（外れ値寄りノイズ）
EDGE_BG_QBIN_PACK_R_MUL = 10000    # 量子化RGBを1次元binへ詰める係数（実装都合）
EDGE_BG_QBIN_PACK_G_MUL = 100

# [B] 色差許容（ユーザーしきい値の反映先）: 調整影響が比較的大きい内部定数
NEW_BG_TOL_MIN = 14.0
NEW_BG_TOL_BASE_MIN = 8.0
NEW_BG_TOL_THRESHOLD_GAIN = 0.55   # ユーザーしきい値 → 色差許容への反映係数
NEW_BG_TOL_P98_CAP = 110.0         # 外周ノイズ由来の過大許容を抑える上限
NEW_BG_TOL_P98_MARGIN = 4.0        # p98に足す安全マージン
NEW_BG_TOL_MAX = 160.0             # 異常に広い許容にならないための最終上限

# [C] 内容帯抽出（色差プロファイル）: 内部専用ヒューリスティック
NEW_BG_PROFILE_SMOOTH_DIVISOR = 48     # 画像サイズ→平滑化窓幅への変換係数
NEW_BG_PROFILE_SMOOTH_MIN_WINDOW = 5   # 平滑化窓の最小幅（奇数になる）
NEW_BG_PROFILE_PERCENTILE = 95.0       # 行/列プロファイルに使う色差percentile
NEW_BG_PROFILE_MASK_LOW_PERCENTILE = 5.0   # プロファイル下限基準
NEW_BG_PROFILE_MASK_HIGH_PERCENTILE = 95.0 # プロファイル上限基準
NEW_BG_PROFILE_MIN_DYNAMIC_RANGE = 1.0     # 変動幅が小さすぎる時の下限
NEW_BG_PROFILE_LEVEL_MIN = 6.0             # 内容帯判定レベルの最低差分
NEW_BG_PROFILE_DR_GAIN = 0.14              # プロファイル変動幅→判定レベルへの反映係数
NEW_BG_MIN_FG_RATIO_FLOOR = 0.006          # 前景率しきい値の下限（細線救済）

# [D] 粗境界→最終境界の補正: 内部専用ヒューリスティック
NEW_BG_MIN_IMAGE_EDGE_PX = 5           # 背景色ベース検出を試す最小辺長
NEW_BG_ROUGH_EDGE_TOUCH_TOL_PX = 1     # 粗境界が端に接しているとみなす許容px
NEW_BG_BLEND_TOL_MIN_PX = 8            # 粗境界とのズレ許容の最小px
NEW_BG_BLEND_TOL_RATIO = 0.16          # 粗境界とのズレ許容の比率（画像辺長基準）
NEW_BG_SIDE_PEEL_MAX_RATIO = 0.12      # 端の低前景帯を剥がす最大割合
NEW_BG_SIDE_PEEL_MIN_FG_FLOOR = 0.003  # 端剥がし時の前景率しきい値の下限
NEW_BG_SIDE_PEEL_MIN_FG_RATIO_GAIN = 0.5  # min_foreground_ratio の端剥がし用縮小率
PEEL_PROFILE_WINDOW = 3                # 端剥がし判定の移動窓幅（小さいほど局所追従）

# 従来ロジック後段の暗縁詰め補助（黒背景前提 / 内部専用）
BLACK_EDGE_BIAS_MAX_PEEL_RATIO = 0.07
BLACK_EDGE_BIAS_MAX_PEEL_PX = 36

# =========================
# 起動時ライブラリチェック
# =========================
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _TK_IMPORT_ERROR = None
except Exception as e:
    # tkinter import失敗は環境依存が大きく例外型が広いため、起動時に包括して捕捉
    tk = None
    ttk = None
    filedialog = None
    messagebox = None
    _TK_IMPORT_ERROR = e


def _safe_stderr_print(text: str):
    """標準エラー出力が失敗しても例外を外へ出さない最終防御。"""
    try:
        print(str(text), file=sys.stderr)
    except Exception:
        # 標準エラー出力自体が壊れている状況でも二重障害にしない
        return


def _show_startup_library_alert(title: str, message: str, kind: str = "error", parent=None):
    """起動前/起動時のライブラリ不足をアラート表示（失敗時は標準エラーへ出力）"""
    shown = False
    if (tk is not None) and (messagebox is not None):
        temp_root = None
        try:
            if parent is None:
                temp_root = tk.Tk()
                temp_root.withdraw()
                try:
                    temp_root.attributes("-topmost", True)
                except Exception as e:
                    # WM環境/プラットフォーム差で失敗しうるため警告のみ
                    _safe_stderr_print(f"[WARN] topmost設定失敗(起動アラート): {e}")
                parent = temp_root
            if kind == "warning":
                messagebox.showwarning(title, message, parent=parent)
            else:
                messagebox.showerror(title, message, parent=parent)
            shown = True
        except Exception:
            # 起動前アラートは補助導線。失敗時はstderrへフォールバック
            shown = False
        finally:
            if temp_root is not None:
                try:
                    temp_root.destroy()
                except Exception as e:
                    # 破棄失敗は終了時の補助処理なので外へ出さない
                    _safe_stderr_print(f"[WARN] 一時Tk破棄失敗(起動アラート): {e}")
    if not shown:
        _safe_stderr_print(f"[{title}]\n{message}")


_STARTUP_REQUIRED_MISSING = []

# 必須ライブラリ
try:
    import numpy as np
except Exception as e:
    np = None
    _STARTUP_REQUIRED_MISSING.append(("numpy", "numpy", str(e)))

try:
    from PIL import Image, ImageTk, ImageDraw
except Exception as e:
    Image = None
    ImageTk = None
    ImageDraw = None
    _STARTUP_REQUIRED_MISSING.append(("Pillow", "Pillow", str(e)))

# 任意ライブラリ（不足しても起動は可能）
_STARTUP_OPTIONAL_MISSING = []

# PDF読み込み用（PyMuPDF）
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    _STARTUP_OPTIONAL_MISSING.append(("PyMuPDF", "pymupdf", "PDF機能"))

# ドラッグ＆ドロップ（任意）
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    DND_FILES = None
    TkinterDnD = None
    _STARTUP_OPTIONAL_MISSING.append(("tkinterdnd2", "tkinterdnd2", "ドラッグ＆ドロップ機能"))

if _TK_IMPORT_ERROR is not None:
    _show_startup_library_alert(
        "起動できません（tkinter不足）",
        "GUIの起動に必要な tkinter を読み込めませんでした。\n"
        "Python の tkinter を有効にした環境で実行してください。\n\n"
        f"詳細: {_TK_IMPORT_ERROR}",
        kind="error",
    )
    raise SystemExit(1)

if _STARTUP_REQUIRED_MISSING:
    missing_lines = []
    install_pkgs = []
    for disp, pkg, err in _STARTUP_REQUIRED_MISSING:
        missing_lines.append(f"- {disp}（pip install {pkg}）")
        install_pkgs.append(pkg)
    install_cmd = "pip install " + " ".join(install_pkgs)
    _show_startup_library_alert(
        "起動できません（ライブラリ不足）",
        "必要なライブラリが不足しているため起動できません。\n\n"
        + "\n".join(missing_lines)
        + "\n\nインストール例:\n"
        + install_cmd
        + "\n\n詳細:\n"
        + "\n".join([f"{disp}: {err}" for disp, _, err in _STARTUP_REQUIRED_MISSING]),
        kind="error",
    )
    raise SystemExit(1)


# =========================
# ユーティリティ
# =========================
def pil_to_rgb(image: Image.Image) -> Image.Image:
    """RGBA等を白背景でRGB化"""
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        return bg
    return image.convert("RGB")


def _smooth_profile_1d(profile, window: int = 9):
    """1次元プロファイルを軽く平滑化（移動平均）"""
    arr = np.asarray(profile, dtype=np.float32)
    n = int(arr.size)
    if n <= 2:
        return arr

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1

    # 長さを超えない最大奇数に調整
    if w > n:
        w = n if (n % 2 == 1) else (n - 1)
    if w < 3:
        return arr

    pad = w // 2
    kernel = np.ones(w, dtype=np.float32) / float(w)
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")



def _contiguous_true_runs(mask):
    """
    bool配列の True 連続区間を [(start, end_exclusive), ...] で返す
    """
    m = np.asarray(mask, dtype=bool)
    n = int(m.size)
    if n == 0:
        return []

    padded = np.concatenate([[False], m, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def _pick_center_content_run(mask):
    """
    True連続区間のうち「長さがある & 中央に近い」区間を選ぶ。
    戻り値: (start, end_exclusive) または None
    """
    runs = _contiguous_true_runs(mask)
    if not runs:
        return None

    n = len(mask)
    center = (n - 1) / 2.0

    best = None
    best_score = -1e18
    for s, e in runs:
        length = e - s
        if length <= 0:
            continue
        run_center = (s + e - 1) / 2.0
        dist = abs(run_center - center)
        # 中央寄りを優先しつつ、十分な長さも評価
        # （外周の細いノイズ線より本文/原稿領域を選びやすくする）
        score = (length * 1.0) - (dist * 0.35)
        if score > best_score:
            best_score = score
            best = (s, e)

    return best


def _detect_non_black_bbox_ratio_fallback(
    img: Image.Image,
    threshold: int = 40,
    padding: int = 0,
    min_non_black_ratio: float = 0.01
):
    """
    フォールバック（v01_06）:
    従来の「非黒画素率」に加え、行/列の高分位明度（p85）も使って
    暗い布地・ノイズで誤って外周を含みやすい問題を抑える。
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)
    if gray.ndim != 2 or gray.size == 0:
        return (0, 0, img.width, img.height)

    # 単純な非黒率（従来）
    non_black = gray > threshold
    row_ratio = non_black.mean(axis=1)
    col_ratio = non_black.mean(axis=0)

    # 暗部ノイズに強い高分位明度プロファイル
    row_p85 = np.percentile(gray, 85, axis=1).astype(np.float32)
    col_p85 = np.percentile(gray, 85, axis=0).astype(np.float32)
    row_sm = _smooth_profile_1d(row_p85, window=max(5, (gray.shape[0] // 50) * 2 + 1))
    col_sm = _smooth_profile_1d(col_p85, window=max(5, (gray.shape[1] // 50) * 2 + 1))

    # 動的しきい値（外周暗部に対して十分明るい行/列を内容候補に）
    def _content_mask(sm, ratio, base_threshold):
        p5 = float(np.percentile(sm, NEW_BG_PROFILE_MASK_LOW_PERCENTILE))
        p95 = float(np.percentile(sm, NEW_BG_PROFILE_MASK_HIGH_PERCENTILE))
        dr = max(NEW_BG_PROFILE_MIN_DYNAMIC_RANGE, p95 - p5)
        bright_thresh = p5 + max(12.0, dr * 0.12)
        # 非黒率も少し使う（細い白地や小さめ原稿の救済）
        ratio_thresh = max(float(min_non_black_ratio), 0.01)
        mask = (sm >= max(base_threshold + 8.0, bright_thresh)) | (ratio >= ratio_thresh * 3.0)
        return mask

    row_mask = _content_mask(row_sm, row_ratio, float(threshold))
    col_mask = _content_mask(col_sm, col_ratio, float(threshold))

    # 中央付近の主要連続区間を採用（外周ノイズを避ける）
    row_run = _pick_center_content_run(row_mask)
    col_run = _pick_center_content_run(col_mask)

    if row_run is None or col_run is None:
        # 旧ロジックへ最終フォールバック
        row_has_content = row_ratio >= min_non_black_ratio
        col_has_content = col_ratio >= min_non_black_ratio
        row_idx = np.flatnonzero(row_has_content)
        col_idx = np.flatnonzero(col_has_content)
        if row_idx.size == 0 or col_idx.size == 0:
            return (0, 0, img.width, img.height)
        top = int(row_idx[0])
        bottom = int(row_idx[-1]) + 1
        left = int(col_idx[0])
        right = int(col_idx[-1]) + 1
    else:
        top, bottom = row_run
        left, right = col_run

    if padding > 0:
        left = max(0, int(left) - padding)
        top = max(0, int(top) - padding)
        right = min(img.width, int(right) + padding)
        bottom = min(img.height, int(bottom) + padding)

    if right <= left or bottom <= top:
        return (0, 0, img.width, img.height)

    return (int(left), int(top), int(right), int(bottom))


def _find_contrast_boundary(profile, side: str, rough_hint: int = None):
    """
    明暗プロファイルの勾配（コントラスト）から外周境界を推定する（v01_06）。
    side: 'top'/'bottom'/'left'/'right'
    rough_hint: 先に求めた概算境界（あればその近傍を優先探索）
    戻り値: 境界インデックス（bboxで使う座標、bottom/rightは終端のexclusive）
    """
    arr = np.asarray(profile, dtype=np.float32)
    n = int(arr.size)
    if n < 8:
        return None

    # サイズに応じて平滑化幅を調整（奇数）
    w = max(5, min(41, (n // 36) * 2 + 1))
    sm = _smooth_profile_1d(arr, window=w)

    grad = np.diff(sm)  # i -> i+1
    if grad.size <= 0:
        return None

    # 外周から最大45%程度を探索（内容内部の線を避けつつ、下辺黒帯にも届くよう拡張）
    search_band = max(8, int(n * 0.45))
    search_band = min(search_band, grad.size)
    if search_band <= 0:
        return None

    p5 = float(np.percentile(sm, 5))
    p95 = float(np.percentile(sm, 95))
    dynamic_range = max(1.0, p95 - p5)
    min_jump = max(6.0, dynamic_range * 0.045)
    min_contrast = max(14.0, dynamic_range * 0.12)
    local_win = max(8, int(n * 0.05))
    local_win = min(local_win, max(8, n // 2))

    def _mean_safe(x):
        return float(np.mean(x)) if x.size > 0 else 0.0

    def _iter_candidate_indices_left(seg):
        # 正の勾配が強い順に候補を試す（単純argmaxより堅牢）
        strong = np.where(seg >= max(min_jump, float(np.percentile(seg, 85))))[0]
        if strong.size == 0:
            strong = np.argsort(seg)[::-1][:12]
        else:
            strong = strong[np.argsort(seg[strong])[::-1][:20]]
        seen = set()
        for i in strong.tolist():
            if i not in seen:
                seen.add(i)
                yield int(i)

    def _iter_candidate_indices_right(seg):
        # 負の勾配（落ち込み）が強い順
        negmag = -seg
        strong = np.where(negmag >= max(min_jump, float(np.percentile(negmag, 85))))[0]
        if strong.size == 0:
            strong = np.argsort(negmag)[::-1][:12]
        else:
            strong = strong[np.argsort(negmag[strong])[::-1][:20]]
        seen = set()
        for i in strong.tolist():
            if i not in seen:
                seen.add(i)
                yield int(i)

    def _score_with_hint(cand):
        if rough_hint is None:
            return 0.0
        return -abs(float(cand) - float(rough_hint)) * 0.15

    if side in ("top", "left"):
        seg = grad[:search_band]
        if seg.size == 0:
            return None

        best = None
        best_score = -1e18

        for rel_idx in _iter_candidate_indices_left(seg):
            jump = float(seg[rel_idx])  # 暗→明 を期待（正）
            cand = rel_idx + 1
            if cand <= 0 or cand >= n:
                continue

            outside = sm[:cand]
            inside = sm[cand:min(n, cand + local_win)]
            if outside.size == 0 or inside.size == 0:
                continue

            outside_mean = _mean_safe(outside)
            inside_mean = _mean_safe(inside)
            contrast = inside_mean - outside_mean
            if jump < min_jump or contrast < min_contrast:
                continue

            # outsideが十分暗く、insideが十分明るい候補を優先
            score = (jump * 1.2) + (contrast * 1.0) + _score_with_hint(cand)
            if score > best_score:
                best_score = score
                best = (cand, outside_mean, inside_mean)

        if best is None:
            return None

        cand, outside_mean, inside_mean = best
        mid = (inside_mean + outside_mean) * 0.5
        s = max(0, cand - local_win)
        e = min(n, cand + local_win)
        for j in range(s, e):
            if sm[j] >= mid and (j == 0 or sm[j - 1] < mid):
                return int(j)
        return int(cand)

    if side in ("bottom", "right"):
        seg = grad[-search_band:]
        if seg.size == 0:
            return None

        best = None
        best_score = -1e18
        base_idx = grad.size - search_band

        for rel_idx in _iter_candidate_indices_right(seg):
            raw_jump = float(seg[rel_idx])  # 明→暗 を期待（負）
            jump = -raw_jump
            idx = base_idx + rel_idx
            cand = idx + 1
            if cand <= 0 or cand >= n:
                continue

            inside = sm[max(0, cand - local_win):cand]
            outside = sm[cand:]
            if inside.size == 0 or outside.size == 0:
                continue

            outside_mean = _mean_safe(outside)
            inside_mean = _mean_safe(inside)
            contrast = inside_mean - outside_mean
            if jump < min_jump or contrast < min_contrast:
                continue

            score = (jump * 1.2) + (contrast * 1.0) + _score_with_hint(cand)
            if score > best_score:
                best_score = score
                best = (cand, outside_mean, inside_mean)

        if best is None:
            return None

        cand, outside_mean, inside_mean = best
        mid = (inside_mean + outside_mean) * 0.5
        s = max(1, cand - local_win)
        e = min(n, cand + local_win)
        for j in range(s, e):
            if sm[j] <= mid and sm[j - 1] > mid:
                return int(j)
        return int(cand)

    return None


def detect_non_black_bbox(
    img: Image.Image,
    threshold: int = 40,
    padding: int = 0,
    min_non_black_ratio: float = 0.01
):
    """
    外周トリム用の境界検出（v01_06）:
    明るい領域と暗い領域のコントラストが大きい位置（勾配が大きい位置）を
    区切り線として優先採用する。
    - まず行/列の高分位明度プロファイルから「内容領域の概算範囲」を求める
    - その近傍でコントラスト境界を探して微調整
    - 失敗時は強化版フォールバックへ
    """
    gray = np.array(img.convert("L"), dtype=np.uint8)

    if gray.ndim != 2 or gray.size == 0:
        return (0, 0, img.width, img.height)

    # 文字や細線の影響を抑えるため高分位（p90）を使用
    row_profile = np.percentile(gray, 90, axis=1).astype(np.float32)
    col_profile = np.percentile(gray, 90, axis=0).astype(np.float32)

    row_sm = _smooth_profile_1d(row_profile, window=max(5, (gray.shape[0] // 48) * 2 + 1))
    col_sm = _smooth_profile_1d(col_profile, window=max(5, (gray.shape[1] // 48) * 2 + 1))

    def _rough_bounds_from_profile(sm, base_thr):
        n = int(sm.size)
        if n <= 0:
            return None

        p5 = float(np.percentile(sm, NEW_BG_PROFILE_MASK_LOW_PERCENTILE))
        p95 = float(np.percentile(sm, NEW_BG_PROFILE_MASK_HIGH_PERCENTILE))
        dr = max(NEW_BG_PROFILE_MIN_DYNAMIC_RANGE, p95 - p5)

        edge_n = max(4, int(n * 0.06))
        edge_vals = np.concatenate([sm[:edge_n], sm[-edge_n:]]) if n >= edge_n * 2 else sm
        edge_dark = float(np.median(edge_vals)) if edge_vals.size > 0 else p5

        # 暗い背景 + 内容の明るさ差を拾う動的しきい値
        thr = max(float(base_thr) + 8.0, edge_dark + max(14.0, dr * 0.12))
        thr = min(thr, p95 - 1.0) if p95 > p5 else thr

        mask = sm >= thr
        run = _pick_center_content_run(mask)
        if run is None:
            return None
        return run

    rough_rows = _rough_bounds_from_profile(row_sm, threshold)
    rough_cols = _rough_bounds_from_profile(col_sm, threshold)

    top_hint = rough_rows[0] if rough_rows else None
    bottom_hint = rough_rows[1] if rough_rows else None
    left_hint = rough_cols[0] if rough_cols else None
    right_hint = rough_cols[1] if rough_cols else None

    top = _find_contrast_boundary(row_profile, "top", rough_hint=top_hint)
    bottom = _find_contrast_boundary(row_profile, "bottom", rough_hint=bottom_hint)
    left = _find_contrast_boundary(col_profile, "left", rough_hint=left_hint)
    right = _find_contrast_boundary(col_profile, "right", rough_hint=right_hint)

    # 強化版フォールバック
    fb_left, fb_top, fb_right, fb_bottom = _detect_non_black_bbox_ratio_fallback(
        img,
        threshold=threshold,
        padding=0,  # 最後にまとめてpadding適用
        min_non_black_ratio=min_non_black_ratio
    )

    # 概算範囲が得られている場合は、コントラスト境界が大きく外れた時に補正
    def _blend_or_fallback(val, fb, rough, side_kind):
        if val is None:
            return int(fb)
        v = int(val)
        if rough is None:
            return v

        # 粗境界から離れすぎる場合はフォールバック/粗境界寄りに寄せる
        tol = max(8, int((gray.shape[0] if side_kind in ("top", "bottom") else gray.shape[1]) * 0.12))
        if abs(v - int(rough)) > tol:
            # どちらが粗境界に近いかを採用
            return int(v if abs(v - int(rough)) <= abs(int(fb) - int(rough)) else int(fb))
        return v

    top = _blend_or_fallback(top, fb_top, top_hint, "top")
    bottom = _blend_or_fallback(bottom, fb_bottom, bottom_hint, "bottom")
    left = _blend_or_fallback(left, fb_left, left_hint, "left")
    right = _blend_or_fallback(right, fb_right, right_hint, "right")

    # 念のための整合性チェック
    top = int(max(0, min(top, img.height)))
    bottom = int(max(0, min(bottom, img.height)))
    left = int(max(0, min(left, img.width)))
    right = int(max(0, min(right, img.width)))

    # 境界が破綻したらフォールバックに戻す
    if right <= left or bottom <= top:
        left, top, right, bottom = fb_left, fb_top, fb_right, fb_bottom

    # サイズが極端に小さい場合もフォールバックに戻す
    if (right - left) < MIN_BBOX_SIZE or (bottom - top) < MIN_BBOX_SIZE:
        left, top, right, bottom = fb_left, fb_top, fb_right, fb_bottom

    # v01_06: 初期bbox内に残った黒帯/暗帯を後段でトリム
    left, top, right, bottom = _refine_bbox_by_dark_edge_trim(
        gray,
        (left, top, right, bottom),
        threshold=threshold,
        max_iter=3,
    )

    # v01_09: 「真っ黒な部分は極力残したくない」方向で、さらに少し内側へ詰める
    # （黒背景が数px残るケース対策。削りすぎ防止の上限付き）
    left, top, right, bottom = _refine_bbox_by_black_edge_bias(
        gray,
        (left, top, right, bottom),
        threshold=threshold,
        max_peel_ratio=BLACK_EDGE_BIAS_MAX_PEEL_RATIO,
        max_peel_px=BLACK_EDGE_BIAS_MAX_PEEL_PX,
    )

    if int(padding) != 0:
        pad = int(padding)
        left = max(0, min(img.width, left - pad))
        top = max(0, min(img.height, top - pad))
        right = max(0, min(img.width, right + pad))
        bottom = max(0, min(img.height, bottom + pad))

    if right <= left or bottom <= top:
        return (0, 0, img.width, img.height)

    return (left, top, right, bottom)



def _estimate_edge_background_rgb(rgb: np.ndarray, base_threshold: int = 40):
    """
    外周帯から背景色を推定する（新ロジック用）。
    - 4辺の帯をサンプル
    - 右/下を少し重み付け（左上に原稿があるケースを避けやすくする）
    戻り値: (bg_rgb_float32[3], info_dict) / 失敗時 (None, {})
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None, {}

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h <= 0 or w <= 0:
        return None, {}

    band = max(2, int(min(h, w) * EDGE_BG_BAND_RATIO))
    band = min(band, max(2, min(h, w) // EDGE_BG_BAND_MAX_DIVISOR))

    strips = []
    try:
        top = rgb[:band, :, :3].reshape(-1, 3)
        bottom = rgb[max(0, h - band):h, :, :3].reshape(-1, 3)
        left = rgb[:, :band, :3].reshape(-1, 3)
        right = rgb[:, max(0, w - band):w, :3].reshape(-1, 3)
        strips.extend([top, bottom, left, right, bottom, right])  # 右/下を優先
    except Exception:
        # 画素統計の取得は探索的処理。失敗時は新ロジックを諦めて呼び出し側でフォールバック
        return None, {}

    edge_pixels = np.concatenate([s for s in strips if s.size > 0], axis=0).astype(np.uint8, copy=False)
    if edge_pixels.size == 0:
        return None, {}

    q_step = EDGE_BG_QUANT_STEP
    q = (edge_pixels // q_step).astype(np.int16)
    qbins = (q[:, 0] * EDGE_BG_QBIN_PACK_R_MUL + q[:, 1] * EDGE_BG_QBIN_PACK_G_MUL + q[:, 2]).astype(np.int32)

    uniq, counts = np.unique(qbins, return_counts=True)
    if uniq.size == 0:
        return None, {}

    best_bin = int(uniq[int(np.argmax(counts))])
    best_ratio = float(np.max(counts)) / float(max(1, qbins.size))
    mask = (qbins == best_bin)

    if np.count_nonzero(mask) < max(EDGE_BG_DOMINANT_MIN_SAMPLES, int(edge_pixels.shape[0] * EDGE_BG_DOMINANT_MIN_RATIO)):
        # 最頻binが弱い場合は全体中央値へフォールバック
        bg = np.median(edge_pixels.astype(np.float32), axis=0)
        sample = edge_pixels.astype(np.float32)
    else:
        sample = edge_pixels[mask].astype(np.float32)
        bg = np.median(sample, axis=0)

    # 背景サンプル揺らぎの目安（色差toleranceの下限調整に使う）
    dev = np.max(np.abs(sample - bg.reshape(1, 3)), axis=1) if sample.size > 0 else np.array([0.0], dtype=np.float32)
    info = {
        "band": int(band),
        "dominant_ratio": float(best_ratio),
        "edge_dev_p90": float(np.percentile(dev, EDGE_BG_DEV_P90_PERCENTILE)) if dev.size > 0 else 0.0,
        "edge_dev_p98": float(np.percentile(dev, EDGE_BG_DEV_P98_PERCENTILE)) if dev.size > 0 else 0.0,
    }
    return bg.astype(np.float32), info


def _detect_non_bgcolor_bbox(
    img: Image.Image,
    threshold: int = 40,
    padding: int = 0,
    min_foreground_ratio: float = 0.01,
):
    """
    新ロジック（背景色ベース / フラットスキャナ蓋なし向け）:
    外周帯から推定した背景色との色差プロファイルで内容bboxを推定する。
    失敗時は None を返し、呼び出し側で従来ロジックへフォールバックする。

    保守メモ:
    - ユーザー設定の影響が大きいのは主に threshold / min_foreground_ratio。
    - その他の NEW_BG_* / EDGE_BG_* は原則「内部専用チューニング定数」として扱う。
    """
    rgb_img = pil_to_rgb(img)
    rgb = np.asarray(rgb_img, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or rgb.size == 0:
        return None

    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h < NEW_BG_MIN_IMAGE_EDGE_PX or w < NEW_BG_MIN_IMAGE_EDGE_PX:
        return None

    bg_rgb, bg_info = _estimate_edge_background_rgb(rgb, base_threshold=threshold)
    if bg_rgb is None:
        return None

    diff = np.abs(rgb.astype(np.int16) - bg_rgb.reshape(1, 1, 3).astype(np.int16))
    dist_max = diff.max(axis=2).astype(np.float32)

    # 背景揺らぎ（edge_dev）+ ユーザーしきい値を合成して色差許容を決める
    # ここが広すぎると黒/灰色背景を内容と誤認、狭すぎると原稿端を欠けやすい。
    edge_dev_p90 = float(bg_info.get("edge_dev_p90", 0.0))
    edge_dev_p98 = float(bg_info.get("edge_dev_p98", edge_dev_p90))
    tol = max(NEW_BG_TOL_MIN, edge_dev_p90 + max(NEW_BG_TOL_BASE_MIN, float(threshold) * NEW_BG_TOL_THRESHOLD_GAIN))
    tol = max(tol, min(NEW_BG_TOL_P98_CAP, edge_dev_p98 + NEW_BG_TOL_P98_MARGIN))
    tol = min(tol, NEW_BG_TOL_MAX)

    fg = dist_max > tol

    # 色差プロファイル（p90/p95）+ 前景率で内容帯を抽出
    row_ratio = fg.mean(axis=1)
    col_ratio = fg.mean(axis=0)

    row_p95 = np.percentile(dist_max, NEW_BG_PROFILE_PERCENTILE, axis=1).astype(np.float32)
    col_p95 = np.percentile(dist_max, NEW_BG_PROFILE_PERCENTILE, axis=0).astype(np.float32)
    row_sm = _smooth_profile_1d(row_p95, window=max(NEW_BG_PROFILE_SMOOTH_MIN_WINDOW, (h // NEW_BG_PROFILE_SMOOTH_DIVISOR) * 2 + 1))
    col_sm = _smooth_profile_1d(col_p95, window=max(NEW_BG_PROFILE_SMOOTH_MIN_WINDOW, (w // NEW_BG_PROFILE_SMOOTH_DIVISOR) * 2 + 1))

    def _content_mask(sm, ratio):
        p5 = float(np.percentile(sm, 5))
        p95 = float(np.percentile(sm, 95))
        dr = max(1.0, p95 - p5)
        level = p5 + max(NEW_BG_PROFILE_LEVEL_MIN, dr * NEW_BG_PROFILE_DR_GAIN)
        ratio_thr = max(float(min_foreground_ratio), NEW_BG_MIN_FG_RATIO_FLOOR)
        return (sm >= level) | (ratio >= ratio_thr)

    row_mask = _content_mask(row_sm, row_ratio)
    col_mask = _content_mask(col_sm, col_ratio)

    row_run = _pick_center_content_run(row_mask)
    col_run = _pick_center_content_run(col_mask)
    if row_run is None or col_run is None:
        return None

    # 粗境界（上左が原稿に接している場合は 0 を維持）
    rough_top, rough_bottom = row_run
    rough_left, rough_right = col_run

    row_profile = row_p95
    col_profile = col_p95

    top = 0 if rough_top <= NEW_BG_ROUGH_EDGE_TOUCH_TOL_PX else _find_contrast_boundary(row_profile, "top", rough_hint=rough_top)
    bottom = h if rough_bottom >= (h - NEW_BG_ROUGH_EDGE_TOUCH_TOL_PX) else _find_contrast_boundary(row_profile, "bottom", rough_hint=rough_bottom)
    left = 0 if rough_left <= NEW_BG_ROUGH_EDGE_TOUCH_TOL_PX else _find_contrast_boundary(col_profile, "left", rough_hint=rough_left)
    right = w if rough_right >= (w - NEW_BG_ROUGH_EDGE_TOUCH_TOL_PX) else _find_contrast_boundary(col_profile, "right", rough_hint=rough_right)

    def _blend_or_rough(val, rough, n):
        if val is None:
            return int(rough)
        v = int(max(0, min(int(val), int(n))))
        tol_px = max(NEW_BG_BLEND_TOL_MIN_PX, int(n * NEW_BG_BLEND_TOL_RATIO))
        if abs(v - int(rough)) > tol_px:
            return int(rough)
        return v

    top = _blend_or_rough(top, rough_top, h)
    bottom = _blend_or_rough(bottom, rough_bottom, h)
    left = _blend_or_rough(left, rough_left, w)
    right = _blend_or_rough(right, rough_right, w)

    # 連続した広い背景に対して外れやすい端を軽く補正（行/列の前景率が低い帯を剥がす）
    try:
        top, bottom = _peel_background_by_profile_ratio(
            row_ratio, top, bottom, side_a="top", side_b="bottom", max_peel_ratio=NEW_BG_SIDE_PEEL_MAX_RATIO, min_fg_ratio=max(NEW_BG_SIDE_PEEL_MIN_FG_FLOOR, min_foreground_ratio * NEW_BG_SIDE_PEEL_MIN_FG_RATIO_GAIN)
        )
        left, right = _peel_background_by_profile_ratio(
            col_ratio, left, right, side_a="left", side_b="right", max_peel_ratio=NEW_BG_SIDE_PEEL_MAX_RATIO, min_fg_ratio=max(NEW_BG_SIDE_PEEL_MIN_FG_FLOOR, min_foreground_ratio * NEW_BG_SIDE_PEEL_MIN_FG_RATIO_GAIN)
        )
    except Exception:
        # プロファイル剥がし失敗時は粗境界を維持（新ロジック内フォールバック）
        top, bottom, left, right = int(top), int(bottom), int(left), int(right)

    if int(padding) != 0:
        pad = int(padding)
        left = max(0, min(w, int(left) - pad))
        top = max(0, min(h, int(top) - pad))
        right = max(0, min(w, int(right) + pad))
        bottom = max(0, min(h, int(bottom) + pad))

    if right <= left or bottom <= top:
        return None
    if (right - left) < MIN_BBOX_SIZE or (bottom - top) < MIN_BBOX_SIZE:
        return None

    return (int(left), int(top), int(right), int(bottom))


def _peel_background_by_profile_ratio(
    profile_ratio,
    start,
    end,
    side_a="top",
    side_b="bottom",
    max_peel_ratio: float = NEW_BG_SIDE_PEEL_MAX_RATIO,
    min_fg_ratio: float = NEW_BG_MIN_FG_RATIO_FLOOR,
):
    """色差前景率プロファイルから、端の低前景帯を少しだけ剥がす（新ロジック補助）"""
    # side_a/side_b はログ/将来拡張用のシグネチャ互換引数（現状の処理分岐では未使用）
    _ = (side_a, side_b)
    arr = np.asarray(profile_ratio, dtype=np.float32)
    n = int(arr.size)
    if n <= 0:
        return int(start), int(end)

    s = int(max(0, min(int(start), n)))
    e = int(max(0, min(int(end), n)))
    if e - s < MIN_BBOX_SIZE:
        return s, e

    max_peel = max(1, int((e - s) * float(max_peel_ratio)))

    # side_a (top/left)
    peeled = 0
    while s < e - MIN_BBOX_SIZE and peeled < max_peel:
        window = arr[s:min(e, s + PEEL_PROFILE_WINDOW)]
        if window.size == 0:
            break
        if float(np.mean(window)) < float(min_fg_ratio):
            s += 1
            peeled += 1
        else:
            break

    # side_b (bottom/right)
    peeled = 0
    while e > s + MIN_BBOX_SIZE and peeled < max_peel:
        window = arr[max(s, e - PEEL_PROFILE_WINDOW):e]
        if window.size == 0:
            break
        if float(np.mean(window)) < float(min_fg_ratio):
            e -= 1
            peeled += 1
        else:
            break

    return int(s), int(e)


def detect_content_bbox(
    img: Image.Image,
    threshold: int = 40,
    padding: int = 0,
    min_non_black_ratio: float = 0.01,
    background_mode: str = "black",
):
    """
    bbox検出の統合入口。
    - background_mode == 'black': 従来ロジック（黒背景前提）
    - background_mode != 'black': 新ロジック（背景色ベース）を優先し、失敗時は従来ロジックへフォールバック
    """
    mode = str(background_mode or "black").strip().lower()
    if mode == "black":
        return detect_non_black_bbox(img, threshold=threshold, padding=padding, min_non_black_ratio=min_non_black_ratio)

    # 新ロジック優先
    try:
        bg_bbox = _detect_non_bgcolor_bbox(img, threshold=threshold, padding=padding, min_foreground_ratio=min_non_black_ratio)
    except Exception:
        # 新ロジック全体の失敗は従来ロジックへフォールバック
        bg_bbox = None

    if bg_bbox is None:
        return detect_non_black_bbox(img, threshold=threshold, padding=padding, min_non_black_ratio=min_non_black_ratio)

    # 新ロジックが全面返しに近い場合は従来も試し、より小さい方を採用
    try:
        full_bbox = (0, 0, int(img.width), int(img.height))
        if tuple(map(int, bg_bbox)) == full_bbox:
            fb = detect_non_black_bbox(img, threshold=threshold, padding=padding, min_non_black_ratio=min_non_black_ratio)
            if tuple(map(int, fb)) != full_bbox:
                return fb
    except Exception:
        # 従来ロジック比較に失敗しても、新ロジック結果は返す
        return bg_bbox

    return bg_bbox



def _refine_bbox_by_dark_edge_trim(
    gray: np.ndarray,
    bbox,
    threshold: int = 40,
    max_iter: int = 3
):
    """
    v01_06:
    初期bboxの内側に残ってしまった黒帯/暗帯を追加で削る後段リファイン。
    「bbox端の行/列が、ほぼ暗部だけで占められている連続帯」を検出して
    端から内側へ詰める。右端/下端の黒布地残り対策に効く。
    """
    if gray.ndim != 2 or gray.size == 0:
        return bbox

    h, w = gray.shape
    l, t, r, b = [int(v) for v in bbox]
    l = max(0, min(l, w))
    r = max(0, min(r, w))
    t = max(0, min(t, h))
    b = max(0, min(b, h))
    if r - l < 5 or b - t < 5:
        return (l, t, r, b)

    # 画像外周（背景が写りやすい）から暗さの基準を推定
    edge = max(2, int(min(h, w) * 0.01))
    edge = min(edge, max(2, min(h, w) // 6))
    if edge * 2 < min(h, w):
        edge_vals = np.concatenate([
            gray[:edge, :].ravel(),
            gray[h-edge:h, :].ravel(),
            gray[:, :edge].ravel(),
            gray[:, w-edge:w].ravel(),
        ])
    else:
        edge_vals = gray.ravel()

    g_p90 = float(np.percentile(gray, 90))
    edge_p50 = float(np.percentile(edge_vals, 50))
    edge_p70 = float(np.percentile(edge_vals, 70))
    edge_p85 = float(np.percentile(edge_vals, 85))
    dr = max(1.0, g_p90 - edge_p50)

    # 暗部しきい値（背景布地の揺らぎに少し余裕）
    dark_thr = max(float(threshold) + 8.0, edge_p70 + max(4.0, dr * 0.06))
    dark_thr = min(dark_thr, edge_p85 + max(6.0, dr * 0.08))
    dark_thr = min(dark_thr, 170.0)  # 明るすぎる背景を暗部扱いしない

    bright_thr = max(dark_thr + 18.0, float(threshold) + 26.0)

    def _edge_run_len(mask_1d, from_start=True):
        n = int(mask_1d.size)
        if n <= 0:
            return 0
        if from_start:
            i = 0
            while i < n and bool(mask_1d[i]):
                i += 1
            return i
        i = n - 1
        c = 0
        while i >= 0 and bool(mask_1d[i]):
            i -= 1
            c += 1
        return c

    def _compute_masks(region, axis_kind):
        # axis_kind: "row" or "col"
        if axis_kind == "row":
            dark_ratio = (region <= dark_thr).mean(axis=1)
            bright_ratio = (region >= bright_thr).mean(axis=1)
            p90 = np.percentile(region, 90, axis=1).astype(np.float32)
            p98 = np.percentile(region, 98, axis=1).astype(np.float32)
        else:
            dark_ratio = (region <= dark_thr).mean(axis=0)
            bright_ratio = (region >= bright_thr).mean(axis=0)
            p90 = np.percentile(region, 90, axis=0).astype(np.float32)
            p98 = np.percentile(region, 98, axis=0).astype(np.float32)

        # 「黒帯らしさ」: ほぼ暗部で、明るい画素が少ない
        bg_like = (
            (dark_ratio >= 0.992)
            | ((dark_ratio >= 0.965) & (p90 <= dark_thr + 7.0) & (bright_ratio <= 0.02))
            | ((dark_ratio >= 0.94) & (p98 <= dark_thr + 10.0) & (bright_ratio <= 0.01))
        )

        # 孤立ノイズで1pxだけ True になるのを少し抑える（1Dの簡易平滑）
        if bg_like.size >= 3:
            sm = bg_like.astype(np.uint8)
            sm2 = sm.copy()
            for i in range(1, len(sm) - 1):
                if sm[i] == 0 and sm[i - 1] == 1 and sm[i + 1] == 1:
                    sm2[i] = 1
                elif sm[i] == 1 and sm[i - 1] == 0 and sm[i + 1] == 0:
                    sm2[i] = 0
            bg_like = sm2.astype(bool)

        return bg_like

    for _ in range(max(1, int(max_iter))):
        changed = False
        if r - l < 5 or b - t < 5:
            break

        region = gray[t:b, l:r]
        hh, ww = region.shape
        if hh < 5 or ww < 5:
            break

        row_bg = _compute_masks(region, "row")
        col_bg = _compute_masks(region, "col")

        # エッジ帯の最小長さ（小さすぎる1-2pxの暗線は誤トリムしにくく）
        min_run_row = max(3, int(hh * 0.006))
        min_run_col = max(3, int(ww * 0.006))

        top_run = _edge_run_len(row_bg, from_start=True)
        bottom_run = _edge_run_len(row_bg, from_start=False)
        left_run = _edge_run_len(col_bg, from_start=True)
        right_run = _edge_run_len(col_bg, from_start=False)

        # 一度に削りすぎない上限（誤検出時の安全装置）
        top_cap = max(4, int(hh * 0.35))
        bottom_cap = max(4, int(hh * 0.35))
        left_cap = max(4, int(ww * 0.35))
        right_cap = max(4, int(ww * 0.35))

        if top_run >= min_run_row:
            trim = min(top_run, top_cap)
            if hh - trim >= 5:
                t += trim
                changed = True

        if bottom_run >= min_run_row and (b - t) > 5:
            trim = min(bottom_run, bottom_cap)
            if (b - t) - trim >= 5:
                b -= trim
                changed = True

        if left_run >= min_run_col:
            trim = min(left_run, left_cap)
            if (r - l) - trim >= 5:
                l += trim
                changed = True

        if right_run >= min_run_col and (r - l) > 5:
            trim = min(right_run, right_cap)
            if (r - l) - trim >= 5:
                r -= trim
                changed = True

        if not changed:
            break

    if r <= l or b <= t:
        return bbox
    return (int(l), int(t), int(r), int(b))



def _refine_bbox_by_black_edge_bias(
    gray: np.ndarray,
    bbox,
    threshold: int = 40,
    max_peel_ratio: float = 0.07,
    max_peel_px: int = 36,
):
    """
    v01_09:
    真っ黒〜濃い暗部の背景が bbox の端に残るケース向けの追加リファイン（強化版）。
    v01_09 より少し攻めて、端の数px〜数十pxを削りにいく。

    方針:
    - 端ライン/薄い帯の「黒っぽさ」を判定（中央領域を重視）
    - 真っ黒判定に加え、濃い暗部（布地/机/影）も削り対象に含める
    - 一度に削りすぎないよう上限を設ける
    """
    if gray.ndim != 2 or gray.size == 0:
        return bbox

    h, w = gray.shape
    l, t, r, b = [int(v) for v in bbox]
    l = max(0, min(l, w))
    r = max(0, min(r, w))
    t = max(0, min(t, h))
    b = max(0, min(b, h))
    if r - l < 5 or b - t < 5:
        return (l, t, r, b)

    region0 = gray[t:b, l:r]
    hh0, ww0 = region0.shape
    if hh0 < 5 or ww0 < 5:
        return (l, t, r, b)

    # bbox端の暗さからローカルなしきい値を作る（グローバル threshold より攻める）
    bw = max(2, min(6, ww0 // 20 if ww0 >= 20 else 2))
    bh = max(2, min(6, hh0 // 20 if hh0 >= 20 else 2))
    edge_vals = np.concatenate([
        region0[:bh, :].ravel(),
        region0[max(0, hh0 - bh):hh0, :].ravel(),
        region0[:, :bw].ravel(),
        region0[:, max(0, ww0 - bw):ww0].ravel(),
    ]) if region0.size > 0 else gray.ravel()

    e50 = float(np.percentile(edge_vals, 50))
    e75 = float(np.percentile(edge_vals, 75))
    e90 = float(np.percentile(edge_vals, 90))
    g95 = float(np.percentile(gray, 95))
    dyn = max(8.0, min(35.0, (g95 - e50) * 0.10))

    # v01_09 は暗部側を広く許容（= より内側まで削りやすい）
    black_thr = min(110.0, max(float(threshold) + 10.0, e75 + 2.0, 26.0))
    dark_thr = min(150.0, max(black_thr + 16.0, e90 + dyn))
    bright_thr = max(155.0, dark_thr + 22.0)
    very_bright_thr = max(180.0, bright_thr + 18.0)

    def _stats_1d(arr1d: np.ndarray):
        x = np.asarray(arr1d, dtype=np.float32)
        if x.size == 0:
            return {
                'very_dark_ratio': 0.0,
                'dark_ratio': 0.0,
                'bright_ratio': 0.0,
                'very_bright_ratio': 0.0,
                'p75': 255.0,
                'p90': 255.0,
                'p95': 255.0,
                'p98': 255.0,
                'med': 255.0,
                'mean': 255.0,
            }
        return {
            'very_dark_ratio': float(np.mean(x <= black_thr)),
            'dark_ratio': float(np.mean(x <= dark_thr)),
            'bright_ratio': float(np.mean(x >= bright_thr)),
            'very_bright_ratio': float(np.mean(x >= very_bright_thr)),
            'p75': float(np.percentile(x, 75)),
            'p90': float(np.percentile(x, 90)),
            'p95': float(np.percentile(x, 95)),
            'p98': float(np.percentile(x, 98)),
            'med': float(np.median(x)),
            'mean': float(np.mean(x)),
        }

    def _line_is_blackish(line_1d: np.ndarray) -> bool:
        if line_1d.size == 0:
            return False
        arr = np.asarray(line_1d, dtype=np.float32)

        # 両端の偶発ノイズを避けるため中央領域を重視
        n = int(arr.size)
        trim = max(0, int(n * 0.08))
        core = arr[trim:n-trim] if (n - 2 * trim) >= 10 else arr

        s_all = _stats_1d(arr)
        s_core = _stats_1d(core)

        # 真っ黒に近い帯（強い判定）
        cond_strict = (
            (s_core['very_dark_ratio'] >= 0.82 and s_core['bright_ratio'] <= 0.010)
            or (s_core['very_dark_ratio'] >= 0.72 and s_core['p95'] <= black_thr + 10.0 and s_core['bright_ratio'] <= 0.006)
            or (s_all['very_dark_ratio'] >= 0.90 and s_all['bright_ratio'] <= 0.010)
        )

        # v01_09: 暗い布地/影まで削るためのやや攻めた判定
        cond_aggressive = (
            (s_core['dark_ratio'] >= 0.92 and s_core['p98'] <= dark_thr + 10.0 and s_core['bright_ratio'] <= 0.008)
            or (s_core['dark_ratio'] >= 0.86 and s_core['p95'] <= dark_thr + 7.0 and s_core['very_bright_ratio'] <= 0.003)
            or (s_core['med'] <= dark_thr - 2.0 and s_core['p90'] <= dark_thr + 8.0 and s_core['bright_ratio'] <= 0.010)
            or (s_all['dark_ratio'] >= 0.95 and s_all['p95'] <= dark_thr + 8.0 and s_all['bright_ratio'] <= 0.008)
        )

        # 白紙の端を誤って削らないガード（明るい画素が一定以上あるなら不可）
        guard_bright = not (
            s_core['bright_ratio'] >= 0.02 or s_core['very_bright_ratio'] >= 0.01 or s_core['p75'] >= bright_thr + 8.0
        )

        return bool((cond_strict or cond_aggressive) and guard_bright)

    # 各辺の最大追加削り量（v01_09より大きめ）
    max_h_peel = max(2, min(int(max_peel_px), int(max(2, (b - t) * float(max_peel_ratio)))))
    max_w_peel = max(2, min(int(max_peel_px), int(max(2, (r - l) * float(max_peel_ratio)))))

    top_used = bottom_used = left_used = right_used = 0

    # 複数辺が連鎖して詰まるよう少し多めにラウンド
    for _ in range(10):
        changed = False
        if r - l < 5 or b - t < 5:
            break

        # 1px判定より安定させるため 3px 帯平均で見る（端のノイズ対策）
        if top_used < max_h_peel and (b - t) > 5:
            strip = gray[t:min(t + 3, b), l:r]
            line = strip.mean(axis=0) if strip.ndim == 2 and strip.shape[0] >= 2 else gray[t, l:r]
            if _line_is_blackish(line):
                t += 1
                top_used += 1
                changed = True

        if bottom_used < max_h_peel and (b - t) > 5:
            strip = gray[max(t, b - 3):b, l:r]
            line = strip.mean(axis=0) if strip.ndim == 2 and strip.shape[0] >= 2 else gray[b - 1, l:r]
            if _line_is_blackish(line):
                b -= 1
                bottom_used += 1
                changed = True

        if left_used < max_w_peel and (r - l) > 5:
            strip = gray[t:b, l:min(l + 3, r)]
            line = strip.mean(axis=1) if strip.ndim == 2 and strip.shape[1] >= 2 else gray[t:b, l]
            if _line_is_blackish(line):
                l += 1
                left_used += 1
                changed = True

        if right_used < max_w_peel and (r - l) > 5:
            strip = gray[t:b, max(l, r - 3):r]
            line = strip.mean(axis=1) if strip.ndim == 2 and strip.shape[1] >= 2 else gray[t:b, r - 1]
            if _line_is_blackish(line):
                r -= 1
                right_used += 1
                changed = True

        if not changed:
            break

    if r <= l or b <= t:
        return bbox
    return (int(l), int(t), int(r), int(b))


def render_pdf_page_to_pil(doc, page_index: int, dpi: int = 144) -> Image.Image:
    """PDFの1ページをPIL画像に変換"""
    if fitz is None:
        raise RuntimeError("PDF処理には PyMuPDF が必要です。pip install pymupdf")
    if doc is None:
        raise RuntimeError("PDFドキュメントが開かれていません。")
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def save_images_as_pdf(images, pdf_path: Path, resolution: float = 144.0):
    """PIL画像のリストをラスタPDFとして保存"""
    if not images:
        raise ValueError("保存する画像がありません。")

    rgb_images = [pil_to_rgb(im) for im in images]
    first = rgb_images[0]
    rest = rgb_images[1:]

    first.save(
        str(pdf_path),
        "PDF",
        save_all=True,
        append_images=rest,
        resolution=float(resolution),
    )


def save_image_files_as_pdf(image_paths, pdf_path: Path, resolution: float = 144.0):
    """画像ファイル群からラスタPDFを保存（読み込み時のハンドルを確実に閉じる）"""
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("保存する画像がありません。")

    opened = []
    try:
        for p in image_paths:
            with Image.open(str(p)) as im:
                opened.append(pil_to_rgb(im.copy()))
        save_images_as_pdf(opened, pdf_path, resolution=resolution)
    finally:
        for im in opened:
            try:
                im.close()
            except (AttributeError, OSError, RuntimeError, ValueError) as e:
                _safe_stderr_print(f"[WARN] PIL画像close失敗(save_image_files_as_pdf): {e}")


def parse_dropped_files(tk_root, data: str):
    """
    tkinterdnd2 の event.data をパス配列に変換
    Windowsの {C:/path with spaces/file.pdf} 形式にも対応
    """
    try:
        paths = list(tk_root.tk.splitlist(data))
    except DND_PARSE_HANDLED_EXCEPTIONS:
        # DnD文字列のsplit失敗時は単一パスとして扱う（入力互換を優先）
        paths = [data]

    cleaned = []
    for p in paths:
        p = p.strip()
        if p.startswith("{") and p.endswith("}"):
            p = p[1:-1]
        p = p.strip('"')
        if p:
            cleaned.append(p)
    return cleaned


# =========================
# アプリ本体
# =========================

_GENERATED_CROP_STEM_RE = re.compile(r"(?i)^.+_crop(?:_p\d{3})?(?:_\d{3})?$")
_GENERATED_CROP_PDF_PAGE_STEM_RE = re.compile(r"(?i)^(?P<base>.+)_crop_p\d{3}(?:_\d{3})?$")
_GENERATED_CROP_GENERIC_STEM_RE = re.compile(r"(?i)^(?P<base>.+)_crop(?:_\d{3})?$")


class CropAppBase:
    SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".pdf"}

    def __init__(self):
        # 現在表示中のファイル/ページ
        self.current_path = None
        self.current_kind = None  # "png" or "pdf"
        self.current_doc = None   # fitz.Document or None
        self.page_count = 0
        self.page_index = 0
        self.preview_pil = None
        self.preview_tk = None
        self.preview_bbox = None

        # 選択キュー（単一ファイルでもここに入れる）
        self.input_queue = []          # list[Path]
        self.queue_index = 0
        self.selection_source = ""     # "file" / "folder"

        # 手動枠（ページ単位）
        # key: (normalized_path_str, page_index) -> (l, t, r, b)
        self.manual_bboxes = {}

        # 自動枠キャッシュ（ページ単位 / プレビュー用）
        # key: (normalized_path_str, page_index) -> {'bbox': (l,t,r,b), 'signature': (...)}
        self.auto_bboxes = {}

        # プレビュー表示の座標変換情報（画像座標 <-> Canvas座標）
        self.canvas_img_x = 0
        self.canvas_img_y = 0
        self.canvas_img_w = 1
        self.canvas_img_h = 1
        self.canvas_scale = 1.0

        # 手動ドラッグ状態
        self.drag_active = False
        self.drag_mode = None          # 'l','r','t','b','lt','rt','lb','rb','move'
        self.drag_start_canvas = (0, 0)  # スクロール考慮後のCanvas座標
        self.drag_start_bbox = None
        self.hit_tolerance_px = 8

        # ページごとの回転（左90°ずつ。0/90/180/270）
        self.page_rotations = {}

        # 表示倍率（fit基準倍率 x ユーザー倍率）
        self.zoom_factor = 1.0
        self.zoom_min = 0.25
        self.zoom_max = 8.0
        self.zoom_step = 1.25
        self.fit_scale = 1.0
        self.zoom_label_var = tk.StringVar(value="表示倍率: 100%")

        self.pdf_preview_dpi = 144     # プレビュー用DPI
        self.pdf_export_dpi = 250      # 出力用DPI

        # 一括処理ワーカー状態
        self.batch_thread = None
        self.batch_running = False
        self.batch_cancel_event = threading.Event()
        self.batch_ui_queue = queue.Queue()
        self._batch_poll_after_id = None

        # 単一出力ワーカー状態（プレビュー系の非同期出力）
        self.single_export_thread = None
        self.single_export_running = False
        self.single_export_cancel_event = threading.Event()
        self._single_export_request = None
        self._single_export_worker_phase = SINGLE_EXPORT_PHASE_IDLE
        self._single_export_worker_src = None

        # UIイベント
        self._resize_after_id = None
        self._live_preview_param_after_id = None
        self._app_closing = False

        # 例外調査用ログ
        self._init_debug_logging()
        self._warn_once_keys = set()
        self._cursor_config_warned = False

        self._build_ui()

    # -------------------------
    # UI構築
    # -------------------------
    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.geometry("1320x820")
        self.root.minsize(980, 720)

        # 上部：選択
        top1 = ttk.Frame(self.root, padding=8)
        top1.pack(fill="x")

        ttk.Button(top1, text="ファイル選択", command=self.choose_file).pack(side="left")
        ttk.Button(top1, text="フォルダ選択", command=self.choose_folder).pack(side="left", padx=(8, 0))
        ttk.Button(top1, text="使い方", command=self._show_usage_help).pack(side="left", padx=(8, 0))

        self.queue_label_var = tk.StringVar(value="対象: 未選択")
        ttk.Label(top1, textvariable=self.queue_label_var).pack(side="left", padx=(16, 0))

        # 上部：設定
        top2 = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        top2.pack(fill="x")

        self.threshold_label = ttk.Label(top2, text="判定しきい値 (0-255):")
        self.threshold_label.pack(side="left")
        self.threshold_var = tk.IntVar(value=40)
        self.threshold_spin = ttk.Spinbox(
            top2,
            from_=0, to=255, width=6, textvariable=self.threshold_var,
            command=self._on_threshold_setting_changed_live,
        )
        self.threshold_spin.pack(side="left", padx=(4, 0))

        ttk.Label(top2, text="背景色:").pack(side="left", padx=(12, 0))
        self.background_mode_var = tk.StringVar(value="black")
        self.background_mode_combo = ttk.Combobox(
            top2,
            width=22,
            state="readonly",
            textvariable=self.background_mode_var,
            values=("黒", "その他（フラットスキャナ蓋無）"),
        )
        self.background_mode_combo.pack(side="left", padx=(4, 0))
        self.background_mode_combo.set("黒")
        self.background_mode_label_map = {
            "black": "黒",
            "other_flatbed_open": "その他（フラットスキャナ蓋無）",
        }
        self._update_threshold_label_for_background_mode()

        ttk.Label(top2, text="余白削除・追加(px):").pack(side="left", padx=(12, 0))
        self.padding_var = tk.IntVar(value=0)
        self.padding_spin = ttk.Spinbox(
            top2,
            from_=-500, to=500, width=6, textvariable=self.padding_var,
            command=self._on_padding_setting_changed_live,
        )
        self.padding_spin.pack(side="left", padx=(4, 0))

        ttk.Label(top2, text="Preview DPI:").pack(side="left", padx=(12, 0))
        self.preview_dpi_var = tk.IntVar(value=self.pdf_preview_dpi)
        self.preview_dpi_spin = ttk.Spinbox(top2, from_=72, to=300, width=5, textvariable=self.preview_dpi_var)
        self.preview_dpi_spin.pack(side="left", padx=(4, 0))

        ttk.Label(top2, text="Export DPI:").pack(side="left", padx=(8, 0))
        self.export_dpi_var = tk.IntVar(value=self.pdf_export_dpi)
        self.export_dpi_spin = ttk.Spinbox(top2, from_=100, to=600, width=5, textvariable=self.export_dpi_var)
        self.export_dpi_spin.pack(side="left", padx=(4, 0))

        self.recalc_btn = ttk.Button(top2, text="再計算", command=self.refresh_preview_bbox)
        self.recalc_btn.pack(side="left", padx=(12, 0))
        self.clear_manual_btn = ttk.Button(top2, text="手動枠を解除", command=self.reset_current_page_bbox_to_auto)
        self.clear_manual_btn.pack(side="left", padx=(12, 0))
        self.export_current_btn = ttk.Button(top2, text="現在ファイルを書き出し", command=self.export_current_only)
        self.export_current_btn.pack(side="left", padx=(8, 0))


        # 実行モード（選択）
        mode_frame = ttk.LabelFrame(self.root, text="処理方法", padding=8)
        mode_frame.pack(fill="x", padx=8, pady=(0, 8))

        self.exec_mode_var = tk.StringVar(value="preview")  # "preview" / "batch"
        ttk.Radiobutton(
            mode_frame, text="プレビューしながら1件ずつ処理", value="preview", variable=self.exec_mode_var
        ).pack(side="left")
        ttk.Radiobutton(
            mode_frame, text="プレビュー無しで全件一括実行", value="batch", variable=self.exec_mode_var
        ).pack(side="left", padx=(16, 0))

        self.start_btn = ttk.Button(mode_frame, text="実行開始", command=self.start_selected_mode)
        self.start_btn.pack(side="left", padx=(20, 0))
        self.cancel_btn = ttk.Button(mode_frame, text="一括処理をキャンセル", command=self.cancel_batch_export)
        self.cancel_btn.pack(side="left", padx=(8, 0))
        self.cancel_btn.state(["disabled"])

        self.export_next_btn = ttk.Button(mode_frame, text="現在を書き出して次へ", command=self.export_current_and_next)
        self.export_next_btn.pack(side="left", padx=(8, 0))
        self.next_file_btn = ttk.Button(mode_frame, text="次のファイルへ", command=self.next_input_file)
        self.next_file_btn.pack(side="left", padx=(8, 0))
        self.prev_file_btn = ttk.Button(mode_frame, text="前のファイルへ", command=self.prev_input_file)
        self.prev_file_btn.pack(side="left", padx=(8, 0))

        # PDFページ移動
        nav = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        nav.pack(fill="x")

        self.prev_btn = ttk.Button(nav, text="◀ 前ページ", command=self.prev_page)
        self.prev_btn.pack(side="left")
        self.next_btn = ttk.Button(nav, text="次ページ ▶", command=self.next_page)
        self.next_btn.pack(side="left", padx=(6, 0))

        self.page_label_var = tk.StringVar(value="ページ: - / -")
        ttk.Label(nav, textvariable=self.page_label_var).pack(side="left", padx=(12, 0))

        self.file_progress_var = tk.StringVar(value="ファイル: - / -")
        ttk.Label(nav, textvariable=self.file_progress_var).pack(side="left", padx=(20, 0))

        self.crop_mode_label_var = tk.StringVar(value="枠: -")
        ttk.Label(nav, textvariable=self.crop_mode_label_var).pack(side="left", padx=(20, 0))

        # プレビュー（スクロールバー付き）
        preview_wrap = ttk.Frame(self.root, padding=8)
        preview_wrap.pack(fill="both", expand=True)

        # プレビュー近傍の表示操作（画像のすぐ近くに配置）
        preview_toolbar = ttk.Frame(preview_wrap, padding=(0, 0, 0, 6))
        preview_toolbar.pack(fill="x")
        ttk.Button(preview_toolbar, text="左回り90°", command=self.rotate_current_left_90).pack(side="left")
        ttk.Button(preview_toolbar, text="拡大", command=self.zoom_in).pack(side="left", padx=(8, 0))
        ttk.Button(preview_toolbar, text="縮小", command=self.zoom_out).pack(side="left", padx=(4, 0))
        ttk.Button(preview_toolbar, text="画面に合わせる", command=self.zoom_fit).pack(side="left", padx=(4, 0))
        ttk.Label(preview_toolbar, textvariable=self.zoom_label_var).pack(side="left", padx=(10, 0))

        preview_canvas_frame = ttk.Frame(preview_wrap)
        preview_canvas_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(preview_canvas_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas_hscroll = ttk.Scrollbar(preview_canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas_vscroll = ttk.Scrollbar(preview_canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(
            xscrollcommand=self.canvas_hscroll.set,
            yscrollcommand=self.canvas_vscroll.set
        )

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas_hscroll.grid(row=1, column=0, sticky="ew")
        preview_canvas_frame.grid_rowconfigure(0, weight=1)
        preview_canvas_frame.grid_columnconfigure(0, weight=1)

        # ステータス
        status = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        status.pack(fill="x")
        self.status_var = tk.StringVar(value="準備完了")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")

        # イベント
        self.root.bind("<Configure>", self._on_resize)
        self.canvas.bind("<Configure>", self._on_resize)
        self.threshold_spin.bind("<Return>", self._on_threshold_setting_changed_live)
        self.threshold_spin.bind("<KeyRelease>", self._on_threshold_setting_changed_live)
        self.threshold_spin.bind("<FocusOut>", self._on_threshold_setting_changed_live)
        self.padding_spin.bind("<Return>", self._on_padding_setting_changed_live)
        self.padding_spin.bind("<KeyRelease>", self._on_padding_setting_changed_live)
        self.padding_spin.bind("<FocusOut>", self._on_padding_setting_changed_live)
        self.preview_dpi_spin.bind("<Return>", lambda e: self._on_preview_dpi_enter())
        self.export_dpi_spin.bind("<Return>", lambda e: self.validate_processing_settings())
        self.background_mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_background_mode_changed())

        # Canvasドラッグ（手動枠調整）
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)

        # DnD設定（利用可能な場合のみ）
        # 画像表示領域（Canvas）へ直接ドロップできるようにする。
        # ※ ルートにも登録しておくと、Canvas外へ少し外しても取りこぼしにくい。
        if DND_AVAILABLE:
            try:
                self.root.drop_target_register(DND_FILES)
                self.root.dnd_bind("<<Drop>>", self._on_drop)

                self.canvas.drop_target_register(DND_FILES)
                self.canvas.dnd_bind("<<Drop>>", self._on_drop)
            except (RuntimeError, ValueError, AttributeError) as e:
                self._log_exception("DnD初期化失敗", exc=e)
                self._set_status(f"DnD初期化に失敗: {e}")

        try:
            self.root.protocol("WM_DELETE_WINDOW", self._on_close_request)
        except TK_UI_HANDLED_EXCEPTIONS as e:
            self._log_debug("WARN", "WM_DELETE_WINDOWフック設定失敗", exc=e)

        self._update_auto_bbox_action_buttons()

    def _show_usage_help(self):
        """使い方ダイアログを表示"""
        help_text = (
            "このアプリは、スキャン画像/PDFの黒枠や余白を自動検出してクロップするGUIツールです。\n\n"
            "【基本の流れ】\n"
            "1. ［ファイル選択］または［フォルダ選択］で対象を読み込みます。\n"
            "2. プレビューの赤枠（クロップ範囲）を確認します。\n"
            "3. 必要に応じて、しきい値・背景色モード・余白削除・追加(px)を調整します。\n"
            "4. うまく切れない場合は、プレビュー上で赤枠をドラッグして手動調整します。\n"
            "5. ［現在ファイルを書き出し］または［現在を書き出して次へ］、一括実行を使います。\n\n"
            "【設定のポイント】\n"
            "・背景色モード\n"
            "  - 黒: 真っ黒背景でスキャンしたとき\n"
            "  - その他（フラットスキャナ蓋無）: 蓋を開けたスキャン等で背景がグレーっぽいとき\n"
            "・しきい値 / 背景色許容差\n"
            "  値を変えるとプレビューにライブ反映されます（入力中は少し待って反映）。\n"
            "・余白削除・追加(px)\n"
            "  +値: 余白を追加（広げる） / -値: 余白を削除（内側に詰める）\n"
            "  こちらもプレビューにライブ反映されます。\n\n"
            "【手動調整】\n"
            "・プレビュー上の赤枠の辺/角をドラッグして微調整できます。\n"
            "・［手動枠を解除］で自動判定の枠に戻せます。\n"
            "・手動調整済みのページでは、しきい値や余白変更時に手動枠を維持します。\n\n"
            "【操作補助】\n"
            "・左回り90°: スキャン向きが違うときに使用\n"
            "・拡大 / 縮小 / 画面に合わせる: 枠調整時の見やすさ改善\n"
            "・スクロールバー: 拡大時の表示移動\n\n"
            "【処理方法】\n"
            "・プレビューしながら1件ずつ処理: 1ページずつ確認したいとき\n"
            "・プレビュー無しで全件一括実行: 設定が決まった後にまとめて処理\n\n"
            "【うまくいかないとき】\n"
            "・背景色モードを切り替える\n"
            "・しきい値（背景色許容差）を調整する\n"
            "・余白削除・追加(px)を調整する\n"
            "・数ページで設定を決めてから一括実行する\n"
        )
        try:
            messagebox.showinfo("使い方", help_text, parent=self.root)
        except TK_UI_HANDLED_EXCEPTIONS:
            messagebox.showinfo("使い方", help_text)

    # -------------------------
    # キー/状態管理（手動枠）
    # -------------------------
    def _normalize_path_key(self, path: Path) -> str:
        try:
            return str(Path(path).resolve())
        except PATH_NORMALIZE_HANDLED_EXCEPTIONS:
            return str(path)

    def _current_page_key(self):
        if self.current_path is None:
            return None
        pkey = self._normalize_path_key(self.current_path)
        page = 0 if self.current_kind == "png" else int(self.page_index)
        return (pkey, page)

    def _get_manual_bbox_for_current_page(self):
        key = self._current_page_key()
        if key is None:
            return None
        return self.manual_bboxes.get(key)

    def _set_manual_bbox_for_current_page(self, bbox):
        key = self._current_page_key()
        if key is not None:
            self.manual_bboxes[key] = tuple(map(int, bbox))

    def _clear_manual_bbox_for_current_page(self):
        key = self._current_page_key()
        if key is not None and key in self.manual_bboxes:
            del self.manual_bboxes[key]

    def _has_manual_bbox_for_current_page(self):
        key = self._current_page_key()
        return key in self.manual_bboxes if key else False

    def _get_auto_bbox_signature_for_current_page(self):
        """現在のプレビュー画像に対する自動枠キャッシュの有効性判定用シグネチャ"""
        if self.preview_pil is None:
            return None
        try:
            threshold = int(self.get_threshold_safe())
        except COERCE_HANDLED_EXCEPTIONS:
            threshold = 40
        try:
            background_mode = str(self.get_background_mode_safe())
        except COERCE_HANDLED_EXCEPTIONS:
            background_mode = "black"
        try:
            padding = int(self.get_padding_safe())
        except COERCE_HANDLED_EXCEPTIONS:
            padding = 0
        try:
            rotation = int(self._get_rotation_for_current_page()) % 360
        except COERCE_HANDLED_EXCEPTIONS:
            rotation = 0
        return (threshold, background_mode, padding, int(self.preview_pil.width), int(self.preview_pil.height), rotation)

    def _set_cached_auto_bbox_for_current_page(self, bbox):
        key = self._current_page_key()
        sig = self._get_auto_bbox_signature_for_current_page()
        if key is None or sig is None or bbox is None:
            return
        try:
            norm = tuple(map(int, bbox))
        except (TypeError, ValueError, AttributeError):
            return
        self.auto_bboxes[key] = {"bbox": norm, "signature": sig}

    def _get_cached_auto_bbox_for_current_page(self):
        key = self._current_page_key()
        sig = self._get_auto_bbox_signature_for_current_page()
        if key is None or sig is None:
            return None
        entry = self.auto_bboxes.get(key)
        if not entry:
            return None
        if tuple(entry.get("signature", ())) != tuple(sig):
            return None
        bbox = entry.get("bbox")
        if bbox is None:
            return None
        try:
            return tuple(map(int, bbox))
        except COERCE_HANDLED_EXCEPTIONS:
            return None

    def _update_auto_bbox_action_buttons(self):
        has_preview = (self.preview_pil is not None and self.current_path is not None)
        has_manual = has_preview and self._has_manual_bbox_for_current_page()

        try:
            if hasattr(self, "clear_manual_btn") and self.clear_manual_btn is not None:
                self.clear_manual_btn.state(["!disabled"] if has_manual else ["disabled"])
        except TK_UI_HANDLED_EXCEPTIONS as e:
            self._log_warn_once("clear_manual_btn_state", "手動枠解除ボタンの状態更新に失敗", exc=e)

    def _set_crop_mode_label(self):
        if self.current_path is None or self.preview_bbox is None:
            self.crop_mode_label_var.set("枠: -")
            return
        self.crop_mode_label_var.set("枠: 手動" if self._has_manual_bbox_for_current_page() else "枠: 自動")

    def _snapshot_export_state_for_paths(self, paths):
        """出力開始時の手動枠/回転状態をスナップショット化（ワーカー中のUI変更影響を防ぐ）。"""
        path_keys = set()
        for p in (paths or []):
            try:
                path_keys.add(self._normalize_path_key(Path(p)))
            except PATH_NORMALIZE_HANDLED_EXCEPTIONS:
                continue

        manual_snapshot = {}
        rotation_snapshot = {}

        try:
            manual_items = list(getattr(self, "manual_bboxes", {}).items())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            manual_items = []
        for key, bbox in manual_items:
            try:
                pkey, page_idx = key
            except (TypeError, ValueError):
                continue
            if pkey not in path_keys:
                continue
            try:
                manual_snapshot[(str(pkey), int(page_idx))] = tuple(map(int, bbox))
            except (TypeError, ValueError):
                continue

        try:
            rotation_items = list(getattr(self, "page_rotations", {}).items())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            rotation_items = []
        for key, angle in rotation_items:
            try:
                pkey, page_idx = key
            except (TypeError, ValueError):
                continue
            if pkey not in path_keys:
                continue
            try:
                rotation_snapshot[(str(pkey), int(page_idx))] = int(angle) % 360
            except (TypeError, ValueError):
                continue

        return {
            "manual_bboxes": manual_snapshot,
            "page_rotations": rotation_snapshot,
        }

    def _prune_page_state_caches(self, keep_paths=None, *, keep_current: bool = True):
        """キュー更新後にページ状態キャッシュを整理（長時間運用での残留を抑制）。"""
        keep_keys = set()
        for p in (keep_paths or []):
            try:
                keep_keys.add(self._normalize_path_key(Path(p)))
            except PATH_NORMALIZE_HANDLED_EXCEPTIONS:
                continue
        if keep_current and getattr(self, "current_path", None) is not None:
            try:
                keep_keys.add(self._normalize_path_key(Path(self.current_path)))
            except PATH_NORMALIZE_HANDLED_EXCEPTIONS:
                pass

        def _prune_map(src_map):
            pruned = {}
            try:
                items = list((src_map or {}).items())
            except INFRA_HANDLED_EXCEPTIONS:
                return {}
            for key, value in items:
                try:
                    pkey = key[0]
                except (TypeError, ValueError, IndexError):
                    continue
                if pkey in keep_keys:
                    pruned[key] = value
            return pruned

        try:
            self.manual_bboxes = _prune_map(getattr(self, "manual_bboxes", {}))
            self.auto_bboxes = _prune_map(getattr(self, "auto_bboxes", {}))
            self.page_rotations = _prune_map(getattr(self, "page_rotations", {}))
        except INFRA_HANDLED_EXCEPTIONS as e:
            self._log_debug("WARN", "ページ状態キャッシュ整理失敗", exc=e)

    # -------------------------
    # ページ回転 / 表示倍率
    # -------------------------
    def _get_page_key(self, path: Path, page_index: int):
        return (self._normalize_path_key(path), int(page_index))

    def _get_rotation_for_page(self, path: Path, page_index: int) -> int:
        key = self._get_page_key(path, page_index)
        return int(self.page_rotations.get(key, 0)) % 360

    def _get_rotation_for_page_with_export_state(self, path: Path, page_index: int, export_state=None) -> int:
        if export_state:
            try:
                key = (self._normalize_path_key(path), int(page_index))
                rot_map = dict(export_state.get("page_rotations") or {})
                return int(rot_map.get(key, 0)) % 360
            except (TypeError, ValueError, AttributeError):
                pass
        return self._get_rotation_for_page(path, page_index)

    def _get_rotation_for_current_page(self) -> int:
        if self.current_path is None:
            return 0
        page = 0 if self.current_kind == "png" else int(self.page_index)
        return self._get_rotation_for_page(self.current_path, page)

    def _set_rotation_for_current_page(self, angle_deg: int):
        if self.current_path is None:
            return
        page = 0 if self.current_kind == "png" else int(self.page_index)
        key = self._get_page_key(self.current_path, page)
        angle = int(angle_deg) % 360
        if angle == 0:
            self.page_rotations.pop(key, None)
        else:
            self.page_rotations[key] = angle

    def _rotate_image_by_page_setting(self, img: Image.Image, src_path: Path, page_index: int, export_state=None) -> Image.Image:
        angle = self._get_rotation_for_page_with_export_state(src_path, page_index, export_state=export_state)
        return self._rotate_image_ccw(img, angle)

    def _rotate_image_ccw(self, img: Image.Image, angle: int) -> Image.Image:
        angle = int(angle) % 360
        if angle == 0:
            return img
        if angle == 90:
            return img.transpose(Image.Transpose.ROTATE_90)
        if angle == 180:
            return img.transpose(Image.Transpose.ROTATE_180)
        if angle == 270:
            return img.transpose(Image.Transpose.ROTATE_270)
        return img.rotate(angle, expand=True)

    def rotate_current_left_90(self):
        if self.current_path is None:
            messagebox.showwarning("未選択", "先にファイルまたはフォルダを選択してください。")
            return
        new_angle = (self._get_rotation_for_current_page() + 90) % 360
        self._set_rotation_for_current_page(new_angle)
        # 回転で寸法が変わるため、現在ページの手動枠は破棄して自動枠を再計算
        self._clear_manual_bbox_for_current_page()
        self._load_current_preview_page()
        self._set_status(f"左回り90°回転: {new_angle}°")

    def _update_zoom_label(self):
        pct = int(round(float(self.zoom_factor) * 100))
        self.zoom_label_var.set(f"表示倍率: {pct}%")

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_max, float(self.zoom_factor) * float(self.zoom_step))
        self._update_zoom_label()
        self._render_preview_to_canvas()

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_min, float(self.zoom_factor) / float(self.zoom_step))
        self._update_zoom_label()
        self._render_preview_to_canvas()

    def zoom_fit(self):
        self.zoom_factor = 1.0
        self._update_zoom_label()
        self._render_preview_to_canvas()

    def _reset_zoom_for_new_page(self):
        self.zoom_factor = 1.0
        self._update_zoom_label()

    def _canvas_event_xy(self, event):
        # スクロール後の座標系（Canvas world座標）へ変換
        return (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))


    def _iter_generated_crop_source_candidates(self, path: Path):
        """生成物らしい名前から、元ファイル候補（同一フォルダ想定）を列挙する。"""
        try:
            p = Path(path)
        except (TypeError, ValueError):
            return []
        stem = str(p.stem)
        parent = p.parent
        current_ext = str(p.suffix)
        candidates = []

        m_pdf_page = _GENERATED_CROP_PDF_PAGE_STEM_RE.match(stem)
        if m_pdf_page:
            base = m_pdf_page.group("base")
            # PDF各ページPNGは元PDFから生成される
            candidates.append(parent / f"{base}.pdf")
            return candidates

        m_generic = _GENERATED_CROP_GENERIC_STEM_RE.match(stem)
        if m_generic:
            base = m_generic.group("base")
            # 単ページ画像/PDF本体の両方の可能性を確認
            if current_ext:
                candidates.append(parent / f"{base}{current_ext}")
            for extra_ext in (".png", ".jpg", ".jpeg", ".pdf"):
                cand = parent / f"{base}{extra_ext}"
                if str(cand) not in {str(x) for x in candidates}:
                    candidates.append(cand)
        return candidates

    def _is_generated_crop_file(self, path: Path, *, require_source_exists: bool = False) -> bool:
        """自アプリ生成物の末尾パターンを判定。必要なら元ファイル存在も確認して誤爆を減らす。"""
        try:
            p = Path(path)
            stem = p.stem
        except (TypeError, ValueError):
            p = None
            stem = str(path)
        if not _GENERATED_CROP_STEM_RE.match(stem):
            return False
        if not require_source_exists:
            return True
        for cand in self._iter_generated_crop_source_candidates(p if p is not None else path):
            try:
                if Path(cand).exists() and Path(cand) != Path(path):
                    return True
            except PATH_NORMALIZE_HANDLED_EXCEPTIONS:
                continue
        return False



    def _init_debug_logging(self):
        self.log_file_path = None
        self.last_error_brief = ""
        try:
            base = Path.home() / ".black_border_cropper"
            base.mkdir(parents=True, exist_ok=True)
            self.log_file_path = base / "black_border_cropper.log"
            self._log_debug("INFO", "アプリ起動", context={"version": APP_VERSION})
        except INFRA_HANDLED_EXCEPTIONS as e:
            self.log_file_path = None
            _safe_stderr_print(f"[LOG-INIT-ERROR] {e}")

    def _short_log_path(self) -> str:
        try:
            return str(self.log_file_path) if self.log_file_path else "標準エラー出力"
        except (AttributeError, TypeError, ValueError):
            return "標準エラー出力"

    def _log_debug(self, level: str, message: str, *, context=None, exc: Exception = None, tb: str = None):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        parts = [f"[{ts}]", f"[{level}]", str(message)]
        if context:
            try:
                ctx_text = ", ".join([f"{k}={v}" for k, v in context.items()])
            except Exception:
                # 任意オブジェクト/壊れたMapping対応のため広く保持（ログ補助処理）
                ctx_text = str(context)
            parts.append(f"| {ctx_text}")
        if exc is not None:
            parts.append(f"| exc={type(exc).__name__}: {exc}")
        line = " ".join(parts)
        _safe_stderr_print(line)
        if tb:
            _safe_stderr_print(tb)
        if self.log_file_path is not None:
            try:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    if tb:
                        f.write(tb.rstrip() + "\n")
            except (OSError, ValueError) as e:
                _safe_stderr_print(f"[WARN] ログファイル書き込み失敗: {e}")

    def _log_exception(self, message: str, *, exc: Exception = None, context=None):
        tb = traceback.format_exc()
        self._log_debug("ERROR", message, context=context, exc=exc, tb=tb)
        self.last_error_brief = f"{message}: {exc}" if exc is not None else message
        return tb

    def _show_error_with_log_hint(self, title: str, user_message: str, *, detail: str = None):
        msg = str(user_message)
        if detail:
            msg += f"\n\n{detail}"
        msg += f"\n\n詳細ログ: {self._short_log_path()}"
        messagebox.showerror(title, msg)

    def _format_dialog_preview_lines(self, items, *, preview_limit: int):
        items = [str(x) for x in (items or [])]
        total = len(items)
        if total <= 0:
            return "-", 0
        preview_n = min(int(preview_limit), total)
        preview_text = "\n".join(items[:preview_n])
        more_count = max(0, total - preview_n)
        return (preview_text + (f"\n... 他 {more_count} 件" if more_count > 0 else "")), total

    def _build_count_summary_lines(self, *, ok_count: int, ng_count: int, skipped: int, total: int = None):
        lines = [f"成功: {int(ok_count)}", f"失敗: {int(ng_count)}", f"スキップ: {int(skipped)}"]
        if total is not None:
            lines.append(f"対象: {int(total)}")
        return "\n".join(lines)

    def _show_export_error(self, file_name: str, result: dict, title: str = "出力失敗"):
        err = (result or {}).get("error") or "不明なエラー"
        tb = (result or {}).get("traceback")
        err_detail = (result or {}).get("error_detail") or {}
        context = {"file": file_name}
        if (result or {}).get("src"):
            context["src"] = (result or {}).get("src")
        if (result or {}).get("kind"):
            context["kind"] = (result or {}).get("kind")
        if err_detail.get("stage"):
            context["stage"] = err_detail.get("stage")
        if err_detail.get("page") is not None:
            context["page"] = err_detail.get("page")
        if tb:
            self._log_debug("ERROR", "出力失敗", context=context, tb=tb)
        else:
            self._log_debug("ERROR", "出力失敗", context=context)
        self._show_error_with_log_hint(title, f"{file_name}\n\n{err}")

    def _build_batch_completion_status(self, *, ok_count: int, ng_count: int, skipped: int, canceled: bool) -> str:
        if canceled:
            return f"一括実行を中断: 成功 {ok_count} / 失敗 {ng_count} / スキップ {skipped}"
        return f"一括実行完了: 成功 {ok_count} / 失敗 {ng_count} / スキップ {skipped}"

    def _show_batch_completion_dialog(self, *, ok_count: int, ng_count: int, skipped: int, total: int, errors, canceled: bool):
        errors = [str(x) for x in (errors or [])]
        summary_lines = self._build_count_summary_lines(ok_count=ok_count, ng_count=ng_count, skipped=skipped, total=total)
        if errors:
            error_preview_text, _ = self._format_dialog_preview_lines(errors, preview_limit=BATCH_ERROR_PREVIEW_LIMIT)
            title = "一括実行中断（失敗あり）" if canceled else "一括実行完了（失敗あり）"
            messagebox.showwarning(title, f"{summary_lines}\n\n{error_preview_text}")
            return

        title = "一括実行中断" if canceled else "一括実行完了"
        messagebox.showinfo(title, summary_lines)

    def _build_single_export_success_dialog(self, result: dict):
        kind = str((result or {}).get("kind") or "").lower()
        pdf_path = (result or {}).get('pdf')
        if kind in (".png", ".jpg", ".jpeg"):
            img_label = (result or {}).get("image_format") or ("PNG" if kind == ".png" else "JPG")
            image_files = (result or {}).get("image_files") or (result or {}).get("pngs") or ["-"]
            preview_text, _ = self._format_dialog_preview_lines(image_files[:1], preview_limit=1)
            return ("出力完了", f"保存しました。\n\nPDF: {pdf_path}\n{img_label}: {preview_text}")

        image_files = (result or {}).get("image_files") or (result or {}).get("pngs") or []
        image_label = (result or {}).get("image_format") or "PNG"
        preview_text, image_count = self._format_dialog_preview_lines(image_files, preview_limit=SINGLE_OUTPUT_PREVIEW_LIMIT)
        return (
            "出力完了",
            f"保存しました。\n\nPDF: {pdf_path}\n{image_label}枚数: {image_count}\n\n{preview_text}",
        )

    def _build_single_export_failure_status(self, action: str) -> str:
        return "出力失敗" if str(action) == "current_and_next" else "出力失敗（現在ファイル）"

    def _resolve_single_export_transition(self, src_str: str):
        """current_and_next 成功後の遷移結果を返す（status/messagebox 用）。"""
        current_str = str(Path(self.current_path)) if self.current_path is not None else None
        if current_str and current_str == str(src_str):
            if self.queue_index < len(self.input_queue) - 1:
                self._open_queue_index(self.queue_index + 1)
                return {
                    "status": f"出力完了: {Path(src_str).name} → 次のファイルを表示",
                    "dialog": None,
                }
            return {
                "status": f"出力完了: {Path(src_str).name}（最後のファイル）",
                "dialog": ("完了", "現在ファイルの出力が完了しました。\nこれで最後のファイルです。"),
            }
        return {
            "status": f"出力完了: {Path(src_str).name}",
            "dialog": ("出力完了", f"{Path(src_str).name} の出力が完了しました。"),
        }

    def _handle_single_export_success_current_only(self, result: dict):
        self._set_status("出力完了（現在ファイル）")
        title, message = self._build_single_export_success_dialog(result)
        messagebox.showinfo(title, message)

    def _handle_single_export_success_current_and_next(self, src_str: str):
        info = self._resolve_single_export_transition(src_str)
        self._set_status(str(info.get("status") or "出力完了"))
        dialog = info.get("dialog")
        if dialog:
            title, message = dialog
            messagebox.showinfo(title, message)

    def _log_warn_once(self, key: str, message: str, *, context=None, exc: Exception = None):
        try:
            if not isinstance(getattr(self, "_warn_once_keys", None), set):
                self._warn_once_keys = set()
            if key in self._warn_once_keys:
                return
            self._warn_once_keys.add(key)
        except INFRA_HANDLED_EXCEPTIONS as e:
            _safe_stderr_print(f"[WARN] warn-once状態更新失敗: {e}")
        self._log_debug("WARN", message, context=context, exc=exc)

    def _safe_close_image(self, img_obj, *, label: str, context=None):
        if img_obj is None:
            return
        try:
            img_obj.close()
        except (AttributeError, OSError, RuntimeError, ValueError) as e:
            self._log_debug("WARN", "PIL画像close失敗", exc=e, context={"label": label, **(context or {})})

    def _set_status(self, text: str):
        try:
            self.status_var.set(text)
        except TK_UI_HANDLED_EXCEPTIONS as e:
            # ステータス更新失敗は致命ではないが、調査できるよう残す
            self._log_debug("WARN", "status更新失敗", exc=e, context={"text": text})

    def _coerce_ui_event_int(self, value, default: int = 0) -> int:
        try:
            return int(value)
        except COERCE_HANDLED_EXCEPTIONS:
            return int(default)

    def _coerce_ui_event_bool(self, value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
        return bool(default)

    def _warn_missing_ui_event_payload_keys(self, event_type: str, payload: dict):
        required = tuple(UI_EVENT_REQUIRED_KEYS.get(str(event_type or ""), ()))
        if not required:
            return
        data = dict(payload or {})
        missing = [k for k in required if k not in data]
        if not missing:
            return
        missing_key = f"ui_event_missing::{event_type}::{'|'.join(missing)}"
        self._log_warn_once(
            missing_key,
            "UIイベント payload の必須キー不足（既定値で補完）",
            context={"event_type": str(event_type), "missing_keys": missing},
        )

    def _normalize_batch_ui_payload_by_type(self, event_type: str, payload: dict):
        """イベント種別ごとの必須キー/型をそろえる（batch/single共通キュー）。

        仕様は `UI_EVENT_REQUIRED_KEYS` を参照。足りないキーはここで既定値補完する。
        """
        event_type = str(event_type or "")
        data = dict(payload or {})
        self._warn_missing_ui_event_payload_keys(event_type, data)
        data["type"] = event_type

        if event_type in (UI_EVENT_STATUS, UI_EVENT_SINGLE_STATUS):
            data["text"] = str(data.get("text") or "")
            return data

        if event_type == UI_EVENT_DONE:
            data["total"] = self._coerce_ui_event_int(data.get("total"), 0)
            data["ok_count"] = self._coerce_ui_event_int(data.get("ok_count"), 0)
            data["ng_count"] = self._coerce_ui_event_int(data.get("ng_count"), 0)
            data["skipped"] = self._coerce_ui_event_int(data.get("skipped"), 0)
            data["canceled"] = self._coerce_ui_event_bool(data.get("canceled"), False)
            try:
                data["errors"] = list(data.get("errors") or [])
            except (TypeError, ValueError):
                data["errors"] = []
            return data

        if event_type == UI_EVENT_SINGLE_DONE:
            data["action"] = str(data.get("action") or "current_only")
            data["src"] = str(data.get("src") or "")
            try:
                data["result"] = dict(data.get("result") or {})
            except (TypeError, ValueError):
                data["result"] = {"ok": False, "error": "invalid single_done payload(result)"}
            return data

        if event_type in (UI_EVENT_FATAL, UI_EVENT_SINGLE_FATAL):
            data["user_message"] = str(data.get("user_message") or "")
            detail = data.get("detail")
            data["detail"] = None if detail is None else str(detail)
            tb = data.get("traceback")
            if tb is not None:
                data["traceback"] = str(tb)
            return data

        return data

    def _set_single_export_worker_phase(self, phase: str, *, src: str = None):
        try:
            self._single_export_worker_phase = str(phase or SINGLE_EXPORT_PHASE_IDLE)
            if src is not None:
                self._single_export_worker_src = str(src)
        except (TypeError, ValueError, AttributeError):
            return

    def _reset_single_export_worker_progress(self):
        self._single_export_worker_phase = SINGLE_EXPORT_PHASE_IDLE
        self._single_export_worker_src = None

    def _build_single_export_close_note(self):
        phase = str(getattr(self, "_single_export_worker_phase", "") or "")
        if phase == SINGLE_EXPORT_PHASE_EXPORTING:
            return "（単一出力は保存/変換中のため、完了待ちになる場合があります）"
        if phase == SINGLE_EXPORT_PHASE_POSTING:
            return "（単一出力は終了処理中です）"
        return ""

    def _build_close_worker_labels(self):
        labels = []
        try:
            if bool(self.batch_thread and self.batch_thread.is_alive()):
                labels.append("一括")
        except THREAD_STATE_HANDLED_EXCEPTIONS:
            pass
        try:
            if bool(self.single_export_thread and self.single_export_thread.is_alive()):
                labels.append("単一出力")
        except THREAD_STATE_HANDLED_EXCEPTIONS:
            pass
        return labels

    def _is_batch_cancel_requested(self) -> bool:
        try:
            return bool(self.batch_cancel_event.is_set())
        except THREAD_STATE_HANDLED_EXCEPTIONS:
            return False

    def _request_batch_cancel(self):
        try:
            self.batch_cancel_event.set()
        except (AttributeError, RuntimeError) as e:
            self._log_debug("WARN", "batch_cancel_event.set失敗", exc=e)

    def _reset_batch_cancel_request(self):
        try:
            if not isinstance(getattr(self, "batch_cancel_event", None), threading.Event):
                self.batch_cancel_event = threading.Event()
            else:
                self.batch_cancel_event.clear()
        except (AttributeError, RuntimeError) as e:
            self._log_debug("WARN", "batch_cancel_event.clear失敗", exc=e)
            self.batch_cancel_event = threading.Event()

    def _is_single_export_cancel_requested(self) -> bool:
        try:
            return bool(self.single_export_cancel_event.is_set())
        except THREAD_STATE_HANDLED_EXCEPTIONS:
            return False

    def _request_single_export_cancel(self):
        try:
            self.single_export_cancel_event.set()
        except (AttributeError, RuntimeError) as e:
            self._log_debug("WARN", "single_export_cancel_event.set失敗", exc=e)

    def _reset_single_export_cancel_request(self):
        try:
            if not isinstance(getattr(self, "single_export_cancel_event", None), threading.Event):
                self.single_export_cancel_event = threading.Event()
            else:
                self.single_export_cancel_event.clear()
        except (AttributeError, RuntimeError) as e:
            self._log_debug("WARN", "single_export_cancel_event.clear失敗", exc=e)
            self.single_export_cancel_event = threading.Event()

    def _schedule_batch_ui_poll(self, delay_ms: int = BATCH_POLL_MS):
        if bool(getattr(self, "_app_closing", False)):
            self._batch_poll_after_id = None
            return False
        try:
            self._batch_poll_after_id = self.root.after(int(delay_ms), self._poll_batch_worker_queue)
            return True
        except TK_UI_HANDLED_EXCEPTIONS as e:
            self._batch_poll_after_id = None
            self._log_debug("WARN", "batch UIポーリング予約失敗", exc=e, context={"delay_ms": int(delay_ms)})
            return False

    def _ensure_batch_ui_polling(self):
        if bool(getattr(self, "_app_closing", False)):
            return False
        if self._batch_poll_after_id is None:
            return self._schedule_batch_ui_poll(BATCH_POLL_MS)
        return True

    def _start_background_worker(self, *, target, args, thread_attr: str, worker_label: str) -> bool:
        try:
            th = threading.Thread(target=target, args=args, daemon=True)
            setattr(self, thread_attr, th)
            th.start()
        except (RuntimeError, TypeError) as e:
            setattr(self, thread_attr, None)
            self._log_exception(f"{worker_label}起動失敗", exc=e)
            self._show_error_with_log_hint(f"{worker_label}起動エラー", f"{worker_label}の開始に失敗しました。", detail=str(e))
            return False
        self._ensure_batch_ui_polling()
        return True

    def _safe_int_from_widget(self, widget, var, *, fallback: int, min_value: int, max_value: int, label: str):
        raw = None
        try:
            raw = widget.get() if widget is not None else var.get()
        except (AttributeError, TypeError, ValueError):
            raw = None

        value = fallback
        valid = True
        try:
            value = int(str(raw).strip())
        except (TypeError, ValueError):
            valid = False
            value = fallback

        clamped = max(int(min_value), min(int(max_value), int(value)))
        if clamped != value:
            valid = False

        try:
            var.set(clamped)
        except TK_UI_HANDLED_EXCEPTIONS as e:
            self._log_warn_once("safe_int_var_set", "数値入力補正のUI反映に失敗", exc=e, context={"label": label, "value": clamped})

        if not valid:
            self._set_status(f"{label} を {clamped} に補正しました")
        return clamped

    def _get_threshold_param_name(self) -> str:
        """背景色モードに応じた、しきい値系パラメータの表示名。"""
        try:
            mode = self.get_background_mode_safe()
        except COERCE_HANDLED_EXCEPTIONS:
            mode = "black"
        return "しきい値" if mode == "black" else "背景色許容差"

    def _get_threshold_label_text(self) -> str:
        name = self._get_threshold_param_name()
        return f"{name} (0-255):"

    def get_threshold_safe(self) -> int:
        return self._safe_int_from_widget(
            self.threshold_spin, self.threshold_var,
            fallback=40, min_value=0, max_value=255, label=self._get_threshold_param_name()
        )

    def get_background_mode_safe(self) -> str:
        try:
            raw = str(self.background_mode_var.get()).strip()
        except COERCE_HANDLED_EXCEPTIONS:
            raw = "黒"

        label_to_mode = {
            "黒": "black",
            "その他（フラットスキャナ蓋無）": "other_flatbed_open",
            "black": "black",
            "other_flatbed_open": "other_flatbed_open",
        }
        mode = label_to_mode.get(raw, "black")

        # UI値が内部コードになっている/不正時は表示を整える
        try:
            if hasattr(self, "background_mode_combo") and self.background_mode_combo is not None:
                wanted_label = "黒" if mode == "black" else "その他（フラットスキャナ蓋無）"
                if str(self.background_mode_var.get()) != wanted_label:
                    self.background_mode_var.set(wanted_label)
        except TK_UI_HANDLED_EXCEPTIONS as e:
            self._log_warn_once("bg_mode_ui_fix", "背景色モード表示補正に失敗", exc=e)

        return mode

    def _update_threshold_label_for_background_mode(self):
        """背景色モードに応じて、しきい値ラベル名をUIで切り替える。"""
        try:
            mode = self.get_background_mode_safe()
        except COERCE_HANDLED_EXCEPTIONS:
            mode = "black"

        text = self._get_threshold_label_text()
        try:
            if hasattr(self, "threshold_label") and self.threshold_label is not None:
                self.threshold_label.configure(text=text)
        except TK_UI_HANDLED_EXCEPTIONS as e:
            self._log_warn_once("threshold_label_update", "しきい値ラベルの更新に失敗", exc=e)

    def get_padding_safe(self) -> int:
        return self._safe_int_from_widget(
            self.padding_spin, self.padding_var,
            fallback=0, min_value=-500, max_value=500, label="余白"
        )

    def get_pdf_preview_dpi_safe(self) -> int:
        value = self._safe_int_from_widget(
            self.preview_dpi_spin, self.preview_dpi_var,
            fallback=int(self.pdf_preview_dpi), min_value=72, max_value=300, label="Preview DPI"
        )
        self.pdf_preview_dpi = int(value)
        return int(value)

    def get_pdf_export_dpi_safe(self) -> int:
        value = self._safe_int_from_widget(
            self.export_dpi_spin, self.export_dpi_var,
            fallback=int(self.pdf_export_dpi), min_value=100, max_value=600, label="Export DPI"
        )
        self.pdf_export_dpi = int(value)
        return int(value)

    def validate_processing_settings(self):
        return {
            "threshold": self.get_threshold_safe(),
            "background_mode": self.get_background_mode_safe(),
            "padding": self.get_padding_safe(),
            "pdf_preview_dpi": self.get_pdf_preview_dpi_safe(),
            "pdf_export_dpi": self.get_pdf_export_dpi_safe(),
        }

    def _on_threshold_setting_changed_live(self, event=None):
        self._on_spinbox_setting_changed_live(
            widget=getattr(self, "threshold_spin", None),
            var=getattr(self, "threshold_var", None),
            reason=self._get_threshold_param_name(),
        )

    def _on_padding_setting_changed_live(self, event=None):
        self._on_spinbox_setting_changed_live(
            widget=getattr(self, "padding_spin", None),
            var=getattr(self, "padding_var", None),
            reason="余白",
        )

    def _on_spinbox_setting_changed_live(self, *, widget, var, reason: str):
        """Spinboxの値変更をプレビューへ即時反映（手動枠は維持）"""
        if self.preview_pil is None or self.current_path is None:
            return

        try:
            raw = widget.get() if widget is not None else (var.get() if var is not None else "")
        except (AttributeError, TypeError, ValueError):
            raw = ""

        raw = str(raw).strip()
        if raw == "":
            return

        try:
            int(raw)
        except (TypeError, ValueError):
            return

        self._schedule_live_auto_bbox_refresh(reason=reason)

    def _schedule_live_auto_bbox_refresh(self, *, reason: str = "設定"):
        if self.preview_pil is None or self.current_path is None:
            return

        try:
            if self._live_preview_param_after_id is not None:
                self.root.after_cancel(self._live_preview_param_after_id)
        except TK_UI_HANDLED_EXCEPTIONS:
            pass

        # 連打/連続入力でも負荷を抑える
        self._live_preview_param_after_id = self.root.after(
            120, lambda: self._run_live_auto_bbox_refresh(reason=reason)
        )

    def _run_live_auto_bbox_refresh(self, *, reason: str = "設定"):
        self._live_preview_param_after_id = None

        if self.preview_pil is None or self.current_path is None:
            return

        # 手動枠は消さない（既存の手動調整を尊重）
        if self._has_manual_bbox_for_current_page():
            self._render_preview_to_canvas()
            self._update_auto_bbox_action_buttons()
            self._set_status(f"{reason}を変更しました（手動枠は維持）")
            return

        try:
            # get_threshold_safe()/get_padding_safe() 側で範囲外入力は補正される
            self.refresh_preview_bbox(update_status=False)
            self._set_status(f"{reason}を変更し、自動枠を更新しました")
        except UI_ACTION_HANDLED_EXCEPTIONS as e:
            self._log_exception(f"{reason}変更時の再計算失敗", exc=e)
            self._show_error_with_log_hint(
                f"{reason}変更エラー",
                f"{reason}変更後の自動枠更新に失敗しました。",
                detail=str(e),
            )

    def _on_background_mode_changed(self):
        # 手動枠編集中は強制変更しない（自動枠時のみ再計算）
        try:
            mode = self.get_background_mode_safe()
            mode_label = "黒" if mode == "black" else "その他（フラットスキャナ蓋無）"
        except COERCE_HANDLED_EXCEPTIONS:
            mode = "black"
            mode_label = "黒"

        self._update_threshold_label_for_background_mode()

        if self.preview_pil is None or self.current_path is None:
            self._set_status(f"背景色モードを {mode_label} に設定しました")
            return

        if self._has_manual_bbox_for_current_page():
            self._set_status(f"背景色モードを {mode_label} に設定しました（手動枠は維持）")
            self._update_auto_bbox_action_buttons()
            return

        try:
            self.refresh_preview_bbox(update_status=False)
            self._set_status(f"背景色モードを {mode_label} に変更し、自動枠を更新しました")
        except UI_ACTION_HANDLED_EXCEPTIONS as e:
            # UIイベント境界（運用時例外はダイアログへ変換し、アプリ継続を優先）
            self._log_exception("背景色モード変更時の再計算失敗", exc=e)
            self._show_error_with_log_hint("背景色モード変更エラー", "背景色モード変更後の自動枠更新に失敗しました。", detail=str(e))

    def _on_preview_dpi_enter(self):
        prev = int(self.pdf_preview_dpi)
        new = self.get_pdf_preview_dpi_safe()
        if new != prev and self.current_kind == "pdf" and self.current_path is not None:
            try:
                self._load_current_preview_page()
                self._set_status(f"Preview DPIを {new} に変更しました")
            except UI_ACTION_HANDLED_EXCEPTIONS as e:
                # UIイベント境界（運用時例外はダイアログへ変換し、アプリ継続を優先）
                self._log_exception("Preview DPI変更時の再読み込み失敗", exc=e, context={"preview_dpi": new})
                self._show_error_with_log_hint("Preview DPI変更エラー", "Preview DPI変更後の再読み込みに失敗しました。", detail=str(e))

    def _next_available_output_path(self, desired_path: Path) -> Path:
        desired_path = Path(desired_path)
        if not desired_path.exists():
            return desired_path
        stem = desired_path.stem
        suffix = desired_path.suffix
        parent = desired_path.parent
        for i in range(1, OUTPUT_NAME_DUPLICATE_MAX + 1):
            candidate = parent / f"{stem}_{i:03d}{suffix}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"出力先が見つかりません: {desired_path}")


    # -------------------------
    # bbox補助
    # -------------------------
    def _clamp_bbox_to_size(self, bbox, width, height, min_size=MIN_BBOX_SIZE):
        if width <= 0 or height <= 0:
            return (0, 0, 1, 1)

        min_w = max(1, min(min_size, width))
        min_h = max(1, min(min_size, height))

        l, t, r, b = [int(round(v)) for v in bbox]

        l = max(0, min(l, width - 1))
        t = max(0, min(t, height - 1))
        r = max(l + min_w, min(r, width))
        b = max(t + min_h, min(b, height))

        # 念のため再調整
        if r > width:
            r = width
            l = max(0, r - min_w)
        if b > height:
            b = height
            t = max(0, b - min_h)

        return (l, t, r, b)

    def _get_auto_bbox_for_image(self, img: Image.Image, threshold: int = None, padding: int = None, background_mode: str = None):
        if threshold is None:
            threshold = self.get_threshold_safe()
        if padding is None:
            padding = self.get_padding_safe()
        if background_mode is None:
            background_mode = self.get_background_mode_safe()
        return detect_content_bbox(
            img,
            threshold=int(threshold),
            padding=int(padding),
            min_non_black_ratio=0.01,
            background_mode=str(background_mode),
        )

    def _redraw_crop_overlay_only(self):
        try:
            self.canvas.delete("crop_overlay")
            self._draw_crop_overlay()
        except Exception as e:
            # 再描画補助の境界。失敗しても本体処理は継続させる（診断ログは残す）
            self._log_debug("WARN", "枠オーバーレイ再描画失敗", exc=e)

    def _apply_preview_bbox_visuals(self, *, redraw_image: bool):
        """bbox反映後の見た目/UI追従をまとめる。
        redraw_image=True: プレビュー全体再描画
        redraw_image=False: 枠オーバーレイのみ更新（ドラッグ中の軽量パス）
        """
        if redraw_image:
            self._render_preview_to_canvas()
        else:
            self._redraw_crop_overlay_only()
        self._set_crop_mode_label()
        self._update_auto_bbox_action_buttons()

    def _build_preview_bbox_status_message(self, bbox, *, manual: bool) -> str:
        if self.preview_pil is None:
            return ""
        l, t, r, b = bbox
        full = (l == 0 and t == 0 and r == self.preview_pil.width and b == self.preview_pil.height)
        mode_text = "手動" if manual or self._has_manual_bbox_for_current_page() else "自動"
        msg = f"{mode_text}枠: left={l}, top={t}, right={r}, bottom={b}"
        if (not manual) and full:
            threshold_name = self._get_threshold_param_name()
            if threshold_name == "しきい値":
                msg += "（画像全体。黒帯を検出できていない可能性あり → しきい値を上げてください）"
            else:
                msg += "（画像全体。背景領域をうまく分離できていない可能性あり → 背景色許容差を調整してください）"
        return msg

    def _apply_preview_bbox(self, bbox, manual=False, update_status=True, redraw_image=True):
        """現在ページのプレビューbboxを適用し、必要なら手動枠として保存。
        - update_status: ステータス欄の更新可否（ドラッグ中は通常 False）
        - redraw_image: True=全体再描画 / False=オーバーレイのみ（軽量）
        """
        if self.preview_pil is None:
            return

        bbox = self._clamp_bbox_to_size(bbox, self.preview_pil.width, self.preview_pil.height)
        self.preview_bbox = bbox

        if manual:
            self._set_manual_bbox_for_current_page(bbox)
        else:
            self._clear_manual_bbox_for_current_page()
            self._set_cached_auto_bbox_for_current_page(bbox)

        self._apply_preview_bbox_visuals(redraw_image=bool(redraw_image))

        if update_status:
            self._set_status(self._build_preview_bbox_status_message(bbox, manual=bool(manual)))

    # -------------------------
    # 選択（ファイル / フォルダ）
    # -------------------------
    def choose_file(self):
        path = filedialog.askopenfilename(
            title="画像（PNG/JPG/JPEG）または PDF を選択",
            filetypes=[
                ("画像 / PDF", "*.png *.jpg *.jpeg *.pdf"),
                ("画像（PNG/JPG/JPEG）", "*.png *.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("JPG / JPEG", "*.jpg *.jpeg"),
                ("PDF", "*.pdf"),
            ],
        )
        if path:
            self.load_queue([Path(path)], selection_source="file")

    def choose_folder(self):
        folder = filedialog.askdirectory(title="フォルダを選択")
        if not folder:
            return
        folder_path = Path(folder)
        paths = self.collect_supported_files_in_folder(folder_path)
        if not paths:
            messagebox.showwarning("対象なし", "フォルダ内に 画像（PNG/JPG/JPEG） / PDF ファイルが見つかりません。")
            return
        self.load_queue(paths, selection_source="folder")

    def collect_supported_files_in_folder(self, folder: Path):
        """フォルダ直下の画像（PNG/JPG/JPEG）/PDFを名前順で取得（非再帰）
        ※ 出力済みの *_crop* ファイルは自動的に除外
        """
        files = []
        for p in sorted(folder.iterdir(), key=lambda x: x.name.lower()):
            if not (p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS):
                continue
            if self._is_generated_crop_file(p, require_source_exists=True):
                continue
            files.append(p)
        return files

    def load_queue(self, paths, selection_source="file"):
        """単一/複数の入力対象をキューとして読み込む（*_crop* は除外）"""
        raw_paths = [Path(p) for p in paths]
        skipped_generated = [p for p in raw_paths if self._is_generated_crop_file(p, require_source_exists=True)]
        self.input_queue = [p for p in raw_paths if not self._is_generated_crop_file(p, require_source_exists=True)]
        self.queue_index = 0
        self.selection_source = selection_source
        self._prune_page_state_caches(self.input_queue, keep_current=False)
        self._update_queue_labels()

        if self.input_queue:
            self._open_queue_index(0)
            if skipped_generated:
                self._set_status(f"選択読み込み: {len(self.input_queue)}件（_crop を {len(skipped_generated)}件スキップ）")
            else:
                self._set_status(f"選択読み込み: {len(self.input_queue)}件")
        else:
            self.clear_preview()
            if skipped_generated:
                self._set_status("対象がありません（*_crop* ファイルは自動スキップ）")
            else:
                self._set_status("対象がありません")

    def _open_queue_index(self, idx: int):
        if not self.input_queue:
            return
        if not (0 <= idx < len(self.input_queue)):
            return
        self.queue_index = idx
        self._open_path_core(self.input_queue[idx])
        self._update_queue_labels()

    def _update_queue_labels(self):
        total = len(self.input_queue)
        if total == 0:
            self.queue_label_var.set("対象: 未選択")
            self.file_progress_var.set("ファイル: - / -")
            return

        source_text = "フォルダ" if self.selection_source == "folder" else "ファイル"
        current_name = self.input_queue[self.queue_index].name if 0 <= self.queue_index < total else "-"
        self.queue_label_var.set(f"対象: {source_text} / {total}件 / 現在: {current_name}")
        self.file_progress_var.set(f"ファイル: {self.queue_index + 1} / {total}")

    def next_input_file(self):
        if not self.input_queue:
            return

        next_idx = self.queue_index + 1
        while next_idx < len(self.input_queue) and self._is_generated_crop_file(self.input_queue[next_idx], require_source_exists=True):
            next_idx += 1

        if next_idx < len(self.input_queue):
            self._open_queue_index(next_idx)
        else:
            self._set_status("最後のファイルです。")

    def prev_input_file(self):
        if not self.input_queue:
            return

        prev_idx = self.queue_index - 1
        while prev_idx >= 0 and self._is_generated_crop_file(self.input_queue[prev_idx], require_source_exists=True):
            prev_idx -= 1

        if prev_idx >= 0:
            self._open_queue_index(prev_idx)
        else:
            self._set_status("最初のファイルです。")

    # -------------------------
    # DnD
    # -------------------------
    def _on_drop(self, event):
        paths = parse_dropped_files(self.root, event.data)
        if not paths:
            return

        resolved = [Path(p) for p in paths if p]
        if not resolved:
            return

        # フォルダが含まれていれば最初のフォルダを優先
        for p in resolved:
            if p.exists() and p.is_dir():
                files = self.collect_supported_files_in_folder(p)
                if not files:
                    messagebox.showwarning("対象なし", "フォルダ内に 画像（PNG/JPG/JPEG） / PDF ファイルが見つかりません。")
                    return
                self.load_queue(files, selection_source="folder")
                return

        # ファイル複数ドロップなら対応拡張子だけをキュー化
        supported_files = [p for p in resolved if p.exists() and p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS]
        if supported_files:
            self.load_queue(supported_files, selection_source="file" if len(supported_files) == 1 else "folder")
            return

        messagebox.showwarning("未対応", "画像（PNG/JPG/JPEG） / PDF ファイル、またはフォルダをドロップしてください。")

    # -------------------------
    # ファイルオープン/プレビュー
    # -------------------------
    def close_current_doc(self):
        if self.current_doc is not None:
            try:
                self.current_doc.close()
            except RuntimeError as e:
                self._log_debug("WARN", "PDF close失敗", exc=e)
        self.current_doc = None

    def clear_preview(self):
        self.close_current_doc()
        self._safe_close_image(self.preview_pil, label="clear_preview", context={"path": str(self.current_path) if self.current_path else None})
        self.current_path = None
        self.current_kind = None
        self.page_count = 0
        self.page_index = 0
        self.preview_pil = None
        self.preview_tk = None
        self.preview_bbox = None
        self.drag_active = False
        self.drag_mode = None
        self._reset_zoom_for_new_page()
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        self.page_label_var.set("ページ: - / -")
        self.crop_mode_label_var.set("枠: -")
        self._update_auto_bbox_action_buttons()
        self._update_queue_labels()

    def _open_path_core(self, path):
        """キュー管理を触らずに単一パスを開く（成功時のみ状態をコミット）"""
        try:
            p = Path(path)
            if not p.exists():
                messagebox.showerror("エラー", f"ファイルが見つかりません:\n{path}")
                return

            ext = p.suffix.lower()
            if ext not in self.SUPPORTED_EXTS:
                messagebox.showerror("エラー", "対応形式は 画像（PNG/JPG/JPEG） / PDF のみです。")
                return

            # 先にローカルで検証・オープンし、成功時のみ状態を反映する
            new_kind = None
            new_doc = None
            new_page_count = 0
            if ext in (".png", ".jpg", ".jpeg"):
                new_kind = "png"  # 単ページ画像（PNG/JPG/JPEG）
                new_page_count = 1
            elif ext == ".pdf":
                if fitz is None:
                    messagebox.showerror(
                        "PyMuPDF未導入",
                        "PDFを扱うには PyMuPDF が必要です。\n\npip install pymupdf"
                    )
                    return
                new_kind = "pdf"
                new_doc = fitz.open(str(p))
                new_page_count = int(new_doc.page_count)
            else:
                messagebox.showerror("エラー", "未対応形式です。")
                return

            old_doc = self.current_doc
            self.current_doc = None
            if old_doc is not None:
                try:
                    old_doc.close()
                except RuntimeError as e:
                    self._log_debug("WARN", "PDF close失敗", exc=e)

            try:
                self.current_path = p
                self.current_kind = new_kind
                self.page_count = new_page_count
                self.page_index = 0
                self.current_doc = new_doc
                self._load_current_preview_page()
            except Exception:
                # 読み込み切替の途中失敗は半更新回避を優先（上位で再通知するため広く保持）
                try:
                    self.clear_preview()
                except (RuntimeError, ValueError):
                    pass
                raise

            self._set_status(f"読み込み完了: {p.name}")
        except UI_ACTION_HANDLED_EXCEPTIONS as e:
            # ファイル読込UI境界（運用時例外は clear_preview で巻き戻して継続）
            self._log_exception("ファイル読み込み失敗", exc=e, context={"path": str(path)})
            self.clear_preview()
            self._show_error_with_log_hint("読み込み失敗", "ファイルの読み込みに失敗しました。", detail=str(e))
            self._set_status("読み込み失敗")

    def _get_current_page_image(self) -> Image.Image:
        if self.current_kind == "png":
            with Image.open(str(self.current_path)) as im:
                return pil_to_rgb(im.copy())
        elif self.current_kind == "pdf":
            dpi = self.get_pdf_preview_dpi_safe()
            return render_pdf_page_to_pil(self.current_doc, self.page_index, dpi=dpi)
        else:
            raise RuntimeError("ファイル未選択")

    def _load_current_preview_page(self):
        if self.current_path is None:
            return

        old_preview = self.preview_pil
        base_img = self._get_current_page_image()
        rotated_img = self._rotate_image_by_page_setting(base_img, self.current_path, 0 if self.current_kind == "png" else int(self.page_index))
        if rotated_img is not base_img:
            self._safe_close_image(base_img, label="preview_base_after_rotate", context={"file": str(self.current_path) if self.current_path else None, "page": int(self.page_index) + 1})
        self.preview_pil = rotated_img
        self._safe_close_image(old_preview, label="preview_replace", context={"file": str(self.current_path) if self.current_path else None})
        self.drag_active = False
        self.drag_mode = None
        self._reset_zoom_for_new_page()

        # 手動枠があれば優先、なければ自動検知
        manual_bbox = self._get_manual_bbox_for_current_page()
        if manual_bbox is not None:
            self.preview_bbox = self._clamp_bbox_to_size(manual_bbox, self.preview_pil.width, self.preview_pil.height)
            self._render_preview_to_canvas()
            self._set_crop_mode_label()
            self._update_auto_bbox_action_buttons()
        else:
            self.refresh_preview_bbox(update_status=False)

        self._update_page_label()

    def _update_page_label(self):
        if self.current_kind == "pdf":
            self.page_label_var.set(f"ページ: {self.page_index + 1} / {self.page_count}")
        elif self.current_kind == "png":
            self.page_label_var.set("ページ: 1 / 1")
        else:
            self.page_label_var.set("ページ: - / -")

        is_pdf = (self.current_kind == "pdf")
        if is_pdf:
            self.prev_btn.state(["!disabled"] if self.page_index > 0 else ["disabled"])
            self.next_btn.state(["!disabled"] if self.page_index < self.page_count - 1 else ["disabled"])
        else:
            self.prev_btn.state(["disabled"])
            self.next_btn.state(["disabled"])

        self._set_crop_mode_label()
        self._update_auto_bbox_action_buttons()

    def prev_page(self):
        if self.current_kind != "pdf":
            return
        if self.page_index > 0:
            self.page_index -= 1
            self._load_current_preview_page()

    def next_page(self):
        if self.current_kind != "pdf":
            return
        if self.page_index < self.page_count - 1:
            self.page_index += 1
            self._load_current_preview_page()

    def refresh_preview_bbox(self, update_status=True):
        """現在ページの自動枠を再計算して適用（手動枠は解除される）"""
        if self.preview_pil is None:
            return
        try:
            bbox = self._get_auto_bbox_for_image(self.preview_pil)
            self._apply_preview_bbox(bbox, manual=False, update_status=update_status)
        except UI_ACTION_HANDLED_EXCEPTIONS as e:
            # UI操作境界（運用時例外はダイアログへ変換して継続）
            self._log_exception("枠再計算失敗", exc=e, context={"file": str(self.current_path) if self.current_path else None})
            self._show_error_with_log_hint("枠再計算エラー", "枠の自動再計算に失敗しました。", detail=str(e))

    def reset_current_page_bbox_to_auto(self):
        """現在ページの手動枠を解除して自動枠に戻す（キャッシュが無ければ再計算）"""
        if self.preview_pil is None:
            return

        if not self._has_manual_bbox_for_current_page():
            self._update_auto_bbox_action_buttons()
            self._set_status("現在ページはすでに自動枠です")
            return

        cached = self._get_cached_auto_bbox_for_current_page()
        try:
            if cached is not None:
                self._apply_preview_bbox(cached, manual=False, update_status=False)
                self._set_status("手動枠を解除して、自動枠（前回計算値）に戻しました")
            else:
                self.refresh_preview_bbox(update_status=False)
                self._set_status("手動枠を解除し、自動枠を再計算して適用しました")
        except UI_ACTION_HANDLED_EXCEPTIONS as e:
            # UI操作境界（運用時例外を通知しつつアプリ継続を優先）
            self._log_exception("手動枠解除失敗", exc=e, context={"file": str(self.current_path) if self.current_path else None})
            self._show_error_with_log_hint("手動枠解除エラー", "手動枠の解除に失敗しました。", detail=str(e))

    # -------------------------
    # Canvas描画（画像 + 手動ドラッグ枠）
    # -------------------------
    def _image_bbox_to_canvas_rect(self, bbox):
        l, t, r, b = bbox
        x1 = self.canvas_img_x + l * self.canvas_scale
        y1 = self.canvas_img_y + t * self.canvas_scale
        x2 = self.canvas_img_x + r * self.canvas_scale
        y2 = self.canvas_img_y + b * self.canvas_scale
        return (x1, y1, x2, y2)

    def _canvas_delta_to_image_delta(self, dx_canvas, dy_canvas):
        s = max(self.canvas_scale, 1e-9)
        return (dx_canvas / s, dy_canvas / s)

    def _draw_crop_overlay(self):
        if self.preview_bbox is None or self.preview_pil is None:
            return

        x1, y1, x2, y2 = self._image_bbox_to_canvas_rect(self.preview_bbox)

        # 枠
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#ff3b30",
            width=2,
            tags=("crop_overlay",)
        )

        # ハンドル（四隅 + 辺中央）
        hs = 4
        handles = [
            (x1, y1), ((x1 + x2) / 2, y1), (x2, y1),
            (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2),
            (x1, y2), ((x1 + x2) / 2, y2), (x2, y2),
        ]
        for hx, hy in handles:
            self.canvas.create_rectangle(
                hx - hs, hy - hs, hx + hs, hy + hs,
                fill="#ff3b30", outline="#ffffff", width=1,
                tags=("crop_overlay",)
            )

        # 補助表示
        mode_text = "MANUAL" if self._has_manual_bbox_for_current_page() else "AUTO"
        self.canvas.create_text(
            x1 + 8, y1 - 8,
            text=mode_text,
            anchor="sw",
            fill="#ff3b30",
            font=("Segoe UI", 9, "bold"),
            tags=("crop_overlay",)
        )

    def _render_preview_to_canvas(self):
        if self.preview_pil is None:
            self.canvas.delete("all")
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            return

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        margin = 20

        img_w, img_h = self.preview_pil.size
        if img_w <= 0 or img_h <= 0:
            return

        avail_w = max(1, canvas_w - margin * 2)
        avail_h = max(1, canvas_h - margin * 2)
        fit_scale_x = avail_w / img_w
        fit_scale_y = avail_h / img_h
        self.fit_scale = max(0.01, min(fit_scale_x, fit_scale_y))

        scale = max(0.01, float(self.fit_scale) * float(self.zoom_factor))

        disp_w = max(1, int(round(img_w * scale)))
        disp_h = max(1, int(round(img_h * scale)))

        # 実際のスケール（丸め後）
        self.canvas_scale = disp_w / img_w
        self.canvas_img_w = disp_w
        self.canvas_img_h = disp_h

        # 画像が小さいときは中央寄せ、大きいときは左上マージン基準
        if disp_w + margin * 2 <= canvas_w:
            self.canvas_img_x = (canvas_w - disp_w) // 2
        else:
            self.canvas_img_x = margin
        if disp_h + margin * 2 <= canvas_h:
            self.canvas_img_y = (canvas_h - disp_h) // 2
        else:
            self.canvas_img_y = margin

        disp_img = self.preview_pil.resize((disp_w, disp_h), Image.LANCZOS)
        self.preview_tk = ImageTk.PhotoImage(disp_img)

        self.canvas.delete("all")
        self.canvas.create_image(self.canvas_img_x, self.canvas_img_y, image=self.preview_tk, anchor="nw")

        scroll_w = max(canvas_w, self.canvas_img_x + disp_w + margin)
        scroll_h = max(canvas_h, self.canvas_img_y + disp_h + margin)
        self.canvas.configure(scrollregion=(0, 0, int(scroll_w), int(scroll_h)))

        # ファイル名表示
        if self.current_path is not None:
            self.canvas.create_text(
                10, 10,
                text=f"{self.current_path.name}",
                anchor="nw",
                fill="white",
                font=("Segoe UI", 10, "bold")
            )

        # 手動調整可能なオーバーレイ枠
        self._draw_crop_overlay()

    def _schedule_preview_render_debounced(self):
        if self.preview_pil is None:
            return
        try:
            if self._resize_after_id is not None:
                self.root.after_cancel(self._resize_after_id)
        except TK_UI_HANDLED_EXCEPTIONS:
            pass
        self._resize_after_id = self.root.after(RESIZE_DEBOUNCE_MS, self._run_debounced_preview_render)

    def _run_debounced_preview_render(self):
        self._resize_after_id = None
        try:
            if self.preview_pil is not None:
                self._render_preview_to_canvas()
        except Exception as e:
            # resize/debounce のUI境界。再描画失敗をログ化してイベントループ継続
            self._log_debug("WARN", "リサイズ後再描画に失敗", exc=e)

    def _on_resize(self, event):
        if self.preview_pil is None:
            return
        if event.widget in (self.root, self.canvas):
            self._schedule_preview_render_debounced()

    # -------------------------
    # Canvasヒットテスト / 手動ドラッグ
    # -------------------------
    def _hit_test_crop_rect(self, x, y):
        """
        クリック位置からドラッグ対象を判定
        戻り値: None / 'l' / 'r' / 't' / 'b' / 'lt' / 'rt' / 'lb' / 'rb' / 'move'
        """
        if self.preview_bbox is None or self.preview_pil is None:
            return None

        x1, y1, x2, y2 = self._image_bbox_to_canvas_rect(self.preview_bbox)
        tol = self.hit_tolerance_px

        near_l = abs(x - x1) <= tol and (y1 - tol) <= y <= (y2 + tol)
        near_r = abs(x - x2) <= tol and (y1 - tol) <= y <= (y2 + tol)
        near_t = abs(y - y1) <= tol and (x1 - tol) <= x <= (x2 + tol)
        near_b = abs(y - y2) <= tol and (x1 - tol) <= x <= (x2 + tol)

        inside = (x1 <= x <= x2 and y1 <= y <= y2)

        # 角優先
        if near_l and near_t:
            return "lt"
        if near_r and near_t:
            return "rt"
        if near_l and near_b:
            return "lb"
        if near_r and near_b:
            return "rb"

        # 辺
        if near_l:
            return "l"
        if near_r:
            return "r"
        if near_t:
            return "t"
        if near_b:
            return "b"

        # 内側は移動
        if inside:
            return "move"

        return None

    def _cursor_for_hit_mode(self, mode):
        # Tk標準カーソル（環境差があるので無難なもの）
        if mode in ("l", "r"):
            return "sb_h_double_arrow"
        if mode in ("t", "b"):
            return "sb_v_double_arrow"
        if mode in ("lt", "rt", "lb", "rb"):
            return "crosshair"
        if mode == "move":
            return "fleur"
        return ""

    def _on_canvas_motion(self, event):
        if self.drag_active:
            return
        cx, cy = self._canvas_event_xy(event)
        mode = self._hit_test_crop_rect(cx, cy)
        try:
            self.canvas.configure(cursor=self._cursor_for_hit_mode(mode))
        except (tk.TclError, RuntimeError, ValueError, AttributeError) as e:
            # Tkカーソル設定は環境差が大きい（OS/テーマ依存）。UI継続を優先して警告のみ
            if not getattr(self, "_cursor_config_warned", False):
                self._cursor_config_warned = True
                self._log_debug("WARN", "カーソル更新に失敗（以降の同種ログは抑制）", exc=e)

    def _on_canvas_press(self, event):
        if self.preview_pil is None or self.preview_bbox is None:
            return
        cx, cy = self._canvas_event_xy(event)
        mode = self._hit_test_crop_rect(cx, cy)
        if mode is None:
            return

        self.drag_active = True
        self.drag_mode = mode
        self.drag_start_canvas = (cx, cy)
        self.drag_start_bbox = tuple(self.preview_bbox)

    def _on_canvas_drag(self, event):
        if not self.drag_active or self.preview_pil is None or self.drag_start_bbox is None:
            return

        cx, cy = self._canvas_event_xy(event)
        start_x, start_y = self.drag_start_canvas
        dx_canvas = cx - start_x
        dy_canvas = cy - start_y
        dx_img, dy_img = self._canvas_delta_to_image_delta(dx_canvas, dy_canvas)

        l0, t0, r0, b0 = self.drag_start_bbox
        mode = self.drag_mode

        img_w, img_h = self.preview_pil.size
        min_w = MIN_BBOX_SIZE
        min_h = MIN_BBOX_SIZE

        if mode == "move":
            w = r0 - l0
            h = b0 - t0
            new_l = int(round(l0 + dx_img))
            new_t = int(round(t0 + dy_img))
            new_l = max(0, min(new_l, img_w - w))
            new_t = max(0, min(new_t, img_h - h))
            new_bbox = (new_l, new_t, new_l + w, new_t + h)
        else:
            l, t, r, b = l0, t0, r0, b0

            if "l" in mode:
                l = int(round(l0 + dx_img))
            if "r" in mode:
                r = int(round(r0 + dx_img))
            if "t" in mode:
                t = int(round(t0 + dy_img))
            if "b" in mode:
                b = int(round(b0 + dy_img))

            # 制約
            l = max(0, min(l, img_w - 1))
            r = max(1, min(r, img_w))
            t = max(0, min(t, img_h - 1))
            b = max(1, min(b, img_h))

            if r - l < min_w:
                if "l" in mode and "r" not in mode:
                    l = r - min_w
                else:
                    r = l + min_w
            if b - t < min_h:
                if "t" in mode and "b" not in mode:
                    t = b - min_h
                else:
                    b = t + min_h

            # 再クランプ
            l = max(0, l)
            t = max(0, t)
            r = min(img_w, r)
            b = min(img_h, b)

            new_bbox = self._clamp_bbox_to_size((l, t, r, b), img_w, img_h, min_size=MIN_BBOX_SIZE)

        # 手動枠として適用
        # ドラッグ中はステータス更新を抑制（リリース時に最終値のみ表示）
        self._apply_preview_bbox(new_bbox, manual=True, update_status=False, redraw_image=False)

    def _on_canvas_release(self, event):
        was_dragging = bool(self.drag_active)
        self.drag_active = False
        self.drag_mode = None
        self.drag_start_bbox = None
        if was_dragging and self.preview_bbox is not None and self.preview_pil is not None:
            try:
                self._apply_preview_bbox(self.preview_bbox, manual=True, update_status=True, redraw_image=False)
            except (RuntimeError, ValueError, TypeError, AttributeError) as e:
                # ドラッグ後の見た目反映は補助処理。失敗時も編集状態は保持して継続
                self._log_debug("WARN", "ドラッグ終了時の最終ステータス更新に失敗", exc=e)

    # -------------------------
    # 画像処理 / 出力
    # -------------------------
    def _resolve_bbox_for_image_export(self, img: Image.Image, src_path: Path, page_index: int, threshold: int = None, padding: int = None, background_mode: str = None, export_state=None):
        """
        出力時に使うbboxを決定（手動枠があればそれを優先）
        """
        key = (self._normalize_path_key(src_path), int(page_index))
        manual = None
        if export_state:
            try:
                manual = dict(export_state.get("manual_bboxes") or {}).get(key)
            except (TypeError, ValueError, AttributeError):
                manual = None
        if manual is None:
            manual = self.manual_bboxes.get(key)

        if manual is not None:
            return self._clamp_bbox_to_size(manual, img.width, img.height)

        return self._get_auto_bbox_for_image(img, threshold=threshold, padding=padding, background_mode=background_mode)

    def _process_single_image(self, img: Image.Image, src_path: Path, page_index: int, threshold: int = None, padding: int = None, background_mode: str = None, export_state=None):
        img = pil_to_rgb(img)
        img = self._rotate_image_by_page_setting(img, src_path=src_path, page_index=page_index, export_state=export_state)
        bbox = self._resolve_bbox_for_image_export(
            img,
            src_path=src_path,
            page_index=page_index,
            threshold=threshold,
            padding=padding,
            background_mode=background_mode,
            export_state=export_state,
        )
        cropped = img.crop(bbox)
        return cropped, bbox

    def _export_image_path(self, src: Path, *, stem: str, out_dir: Path, ext: str, threshold: int, padding: int, background_mode: str, export_dpi: int, cancel_check=None, export_state=None):
        result = {
            "src": str(src),
            "kind": ext,
            "ok": True,
            "pdf": None,
            "pngs": [],
            "image_files": [],
            "image_format": None,
            "error": None,
            "canceled": False,
            "traceback": None,
        }
        processing_stage = "init"
        cropped = None
        try:
            if cancel_check and cancel_check():
                result["ok"] = False
                result["canceled"] = True
                result["error"] = "キャンセルされました"
                return result

            processing_stage = "open_image"
            with Image.open(str(src)) as im:
                processing_stage = "crop_image"
                cropped, _bbox = self._process_single_image(
                    im.copy(),
                    src_path=src,
                    page_index=0,
                    threshold=threshold,
                    padding=padding,
                    background_mode=background_mode,
                    export_state=export_state,
                )

            if cancel_check and cancel_check():
                result["ok"] = False
                result["canceled"] = True
                result["error"] = "キャンセルされました"
                return result

            processing_stage = "prepare_output_paths"
            out_pdf = self._next_available_output_path(out_dir / f"{stem}.pdf")
            if ext == ".png":
                out_img = self._next_available_output_path(out_dir / f"{stem}.png")
                processing_stage = "save_png"
                cropped.save(str(out_img), "PNG")
                result["image_format"] = "PNG"
            else:
                out_img = self._next_available_output_path(out_dir / f"{stem}.jpg")
                processing_stage = "save_jpg"
                cropped.save(str(out_img), "JPEG", quality=JPEG_EXPORT_QUALITY, optimize=True)
                result["image_format"] = "JPG"

            if cancel_check and cancel_check():
                result["ok"] = False
                result["canceled"] = True
                result["error"] = "キャンセルされました"
                try:
                    Path(out_img).unlink(missing_ok=True)
                except OSError:
                    pass
                return result

            processing_stage = "save_pdf"
            save_images_as_pdf([cropped], out_pdf, resolution=float(export_dpi))

            result["pdf"] = str(out_pdf)
            result["image_files"] = [str(out_img)]
            result["pngs"] = [str(out_img)]
            return result
        except EXPORT_HANDLED_EXCEPTIONS as e:
            result["ok"] = False
            result["error"] = f"{e}（処理段階: {processing_stage}）"
            result["traceback"] = traceback.format_exc()
            result["error_detail"] = {
                "stage": processing_stage,
                "page": None,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            }
            self._log_debug("ERROR", "画像出力失敗", exc=e, context={"file": str(src), "stage": processing_stage})
            return result
        finally:
            self._safe_close_image(cropped, label="export_single_image_cropped", context={"file": str(src)})

    def _export_pdf_path(self, src: Path, *, stem: str, out_dir: Path, ext: str, threshold: int, padding: int, background_mode: str, export_dpi: int, cancel_check=None, export_state=None):
        result = {
            "src": str(src),
            "kind": ext,
            "ok": True,
            "pdf": None,
            "pngs": [],
            "image_files": [],
            "image_format": None,
            "error": None,
            "canceled": False,
            "traceback": None,
        }
        processing_stage = "init"
        processing_page = None
        doc = None
        tempdir_ctx = None
        tempdir_path = None
        temp_png_paths = []
        moved_outputs = []
        try:
            if fitz is None:
                raise RuntimeError("PDF処理には PyMuPDF が必要です。pip install pymupdf")

            out_pdf = self._next_available_output_path(out_dir / f"{stem}.pdf")

            processing_stage = "create_tempdir"
            try:
                tempdir_ctx = tempfile.TemporaryDirectory(prefix=f"{stem}__tmp__", dir=str(out_dir))
            except OSError:
                tempdir_ctx = tempfile.TemporaryDirectory(prefix=f"{stem}__tmp__")
            tempdir_path = Path(tempdir_ctx.name)
            compiled_tmp_pdf = tempdir_path / f"{stem}__compiled_tmp.pdf"

            processing_stage = "open_pdf"
            doc = fitz.open(str(src))
            page_count = int(doc.page_count)

            for i in range(page_count):
                processing_page = i + 1
                if cancel_check and cancel_check():
                    result["ok"] = False
                    result["canceled"] = True
                    result["error"] = "キャンセルされました"
                    return result

                img = None
                cropped = None
                try:
                    processing_stage = "render_pdf_page"
                    img = render_pdf_page_to_pil(doc, i, dpi=export_dpi)
                    processing_stage = "crop_pdf_page"
                    cropped, _bbox = self._process_single_image(
                        img,
                        src_path=src,
                        page_index=i,
                        threshold=threshold,
                        padding=padding,
                        background_mode=background_mode,
                        export_state=export_state,
                    )
                    processing_stage = "save_pdf_page_png_temp"
                    tmp_png = tempdir_path / f"page_{i+1:03d}.png"
                    cropped.save(str(tmp_png), "PNG")
                    temp_png_paths.append(tmp_png)
                finally:
                    self._safe_close_image(img, label="export_pdf_page_src", context={"file": str(src), "page": i + 1})
                    self._safe_close_image(cropped, label="export_pdf_page_cropped", context={"file": str(src), "page": i + 1})

            if cancel_check and cancel_check():
                result["ok"] = False
                result["canceled"] = True
                result["error"] = "キャンセルされました"
                return result

            processing_stage = "save_compiled_pdf_temp"
            save_image_files_as_pdf(temp_png_paths, compiled_tmp_pdf, resolution=float(export_dpi))

            if cancel_check and cancel_check():
                result["ok"] = False
                result["canceled"] = True
                result["error"] = "キャンセルされました"
                return result

            processing_stage = "reserve_final_output_paths"
            final_png_paths = []
            for i in range(len(temp_png_paths)):
                final_png_paths.append(self._next_available_output_path(out_dir / f"{stem}_p{i+1:03d}.png"))

            processing_stage = "move_outputs"
            for tmp_png, final_png in zip(temp_png_paths, final_png_paths):
                Path(tmp_png).replace(final_png)
                moved_outputs.append(Path(final_png))
            Path(compiled_tmp_pdf).replace(out_pdf)
            moved_outputs.append(Path(out_pdf))

            result["pdf"] = str(out_pdf)
            result["pngs"] = [str(p) for p in final_png_paths]
            result["image_files"] = [str(p) for p in final_png_paths]
            result["image_format"] = "PNG"
            return result

        except EXPORT_HANDLED_EXCEPTIONS as e:
            for moved in reversed(moved_outputs):
                try:
                    Path(moved).unlink(missing_ok=True)
                except OSError as cleanup_e:
                    self._log_debug("WARN", "PDF出力失敗時の後始末失敗", exc=cleanup_e, context={"path": str(moved), "src": str(src)})

            result["ok"] = False
            stage_text = f"処理段階: {processing_stage}" if processing_stage else ""
            page_text = f" / ページ: {processing_page}" if processing_page is not None else ""
            result["error"] = f"{e}（{stage_text}{page_text}）" if (stage_text or page_text) else str(e)
            result["traceback"] = traceback.format_exc()
            result["error_detail"] = {
                "stage": processing_stage,
                "page": processing_page,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
            }
            self._log_debug(
                "ERROR",
                "PDF出力失敗",
                exc=e,
                context={
                    "file": str(src),
                    "stage": processing_stage,
                    "page": processing_page,
                    "tempdir": str(tempdir_path) if tempdir_path else None,
                },
            )
            return result
        finally:
            if doc is not None:
                try:
                    doc.close()
                except (RuntimeError, OSError, ValueError) as e:
                    self._log_debug("WARN", "PDF close失敗(_export_pdf_path)", exc=e, context={"file": str(src)})
            if tempdir_ctx is not None:
                try:
                    tempdir_ctx.cleanup()
                except OSError as e:
                    self._log_debug("WARN", "一時ディレクトリcleanup失敗", exc=e, context={"file": str(src), "tempdir": str(tempdir_path) if tempdir_path else None})

    def _export_path(self, src: Path, cancel_check=None, settings=None, export_state=None):
        """1ファイルを処理して出力する（settings はUIスナップショットを必須）"""
        src = Path(src)
        stem = src.stem + "_crop"
        out_dir = src.parent
        ext = src.suffix.lower()

        if settings is None:
            raise RuntimeError("内部エラー: _export_path には settings スナップショットが必要です。")

        threshold = int(settings["threshold"])
        background_mode = str(settings.get("background_mode", "black"))
        padding = int(settings["padding"])
        export_dpi = int(settings["pdf_export_dpi"])

        if ext in (".png", ".jpg", ".jpeg"):
            return self._export_image_path(
                src,
                stem=stem,
                out_dir=out_dir,
                ext=ext,
                threshold=threshold,
                padding=padding,
                background_mode=background_mode,
                export_dpi=export_dpi,
                cancel_check=cancel_check,
                export_state=export_state,
            )
        if ext == ".pdf":
            return self._export_pdf_path(
                src,
                stem=stem,
                out_dir=out_dir,
                ext=ext,
                threshold=threshold,
                padding=padding,
                background_mode=background_mode,
                export_dpi=export_dpi,
                cancel_check=cancel_check,
                export_state=export_state,
            )

        return {
            "src": str(src),
            "kind": ext,
            "ok": False,
            "pdf": None,
            "pngs": [],
            "image_files": [],
            "image_format": None,
            "error": "未対応形式です。",
            "canceled": False,
            "traceback": None,
            "error_detail": {
                "stage": "validate_input",
                "page": None,
                "exception_type": "RuntimeError",
                "exception_message": "未対応形式です。",
            },
        }

    def _is_any_worker_running(self) -> bool:
        try:
            return bool(self.batch_running or self.single_export_running)
        except (AttributeError, RuntimeError):
            return bool(self.batch_running)

    def _set_single_export_running_ui(self, running: bool):
        self.single_export_running = bool(running)
        targets = [
            getattr(self, "export_current_btn", None),
            getattr(self, "export_next_btn", None),
            getattr(self, "start_btn", None),
        ]
        for btn in targets:
            if btn is None:
                continue
            try:
                if running:
                    btn.state(["disabled"])
                else:
                    if not self.batch_running:
                        btn.state(["!disabled"])
            except TK_UI_HANDLED_EXCEPTIONS as e:
                self._log_debug("WARN", "単一出力中UI状態更新失敗", exc=e)

    def _single_export_worker_main(self, src_path: Path, settings_snapshot: dict, action: str, export_state_snapshot: dict):
        try:
            self._set_single_export_worker_phase(SINGLE_EXPORT_PHASE_PREPARE, src=str(src_path))
            self._post_batch_ui_event(UI_EVENT_SINGLE_STATUS, text=f"出力中: {Path(src_path).name}")
            self._set_single_export_worker_phase(SINGLE_EXPORT_PHASE_EXPORTING, src=str(src_path))
            result = self._export_path(
                Path(src_path),
                cancel_check=self._is_single_export_cancel_requested,
                settings=settings_snapshot,
                export_state=export_state_snapshot,
            )
            self._set_single_export_worker_phase(SINGLE_EXPORT_PHASE_POSTING, src=str(src_path))
            self._post_batch_ui_event(UI_EVENT_SINGLE_DONE, action=action, src=str(src_path), result=result)
        except Exception as e:
            # ワーカー境界の最終防御（設計上はここで握ってUIへ通知する）
            tb = self._log_exception("単一出力ワーカー致命的エラー", exc=e, context={"src": str(src_path), "action": action})
            self._post_batch_ui_event(
                UI_EVENT_SINGLE_FATAL,
                action=action,
                src=str(src_path),
                user_message="単一ファイル出力中に内部エラーが発生しました。",
                detail=str(e),
                traceback=tb,
            )

    def _start_single_export_worker(self, action: str):
        if self.current_path is None:
            messagebox.showwarning("未選択", "先にファイルまたはフォルダを選択してください。")
            return False
        if self._is_any_worker_running():
            if self.batch_running:
                messagebox.showinfo("実行中", "一括処理を実行中です。停止する場合はキャンセルを押してください。")
            else:
                messagebox.showinfo("実行中", "単一ファイル出力を実行中です。完了を待ってください。")
            return False

        settings_snapshot = dict(self.validate_processing_settings())
        self._reset_single_export_cancel_request()
        src_path = Path(self.current_path)
        export_state_snapshot = self._snapshot_export_state_for_paths([src_path])
        self._single_export_request = {
            "action": str(action),
            "src": str(src_path),
            "manual_bbox_count": len(export_state_snapshot.get("manual_bboxes") or {}),
            "rotation_count": len(export_state_snapshot.get("page_rotations") or {}),
        }
        self._set_single_export_worker_phase(SINGLE_EXPORT_PHASE_PREPARE, src=str(src_path))
        self._set_single_export_running_ui(True)
        self._set_status(f"出力中: {src_path.name}")
        try:
            self.root.update_idletasks()
        except TK_UI_HANDLED_EXCEPTIONS:
            pass

        started = self._start_background_worker(
            target=self._single_export_worker_main,
            args=(src_path, settings_snapshot, str(action), export_state_snapshot),
            thread_attr="single_export_thread",
            worker_label="単一ファイル出力",
        )
        if not started:
            self._set_single_export_running_ui(False)
            self._single_export_request = None
            self._reset_single_export_worker_progress()
            self._set_status("単一ファイル出力を開始できませんでした")
            return False
        return True

    def _handle_single_export_done(self, payload: dict):
        self._set_single_export_running_ui(False)
        self.single_export_thread = None
        self._reset_single_export_worker_progress()
        req = self._single_export_request or {}
        self._single_export_request = None

        action = str(payload.get("action") or req.get("action") or "current_only")
        src_str = str(payload.get("src") or req.get("src") or "")
        src_name = Path(src_str).name if src_str else "現在ファイル"
        result = dict(payload.get("result") or {})

        if bool(result.get("canceled")):
            self._set_status("単一ファイル出力を中止しました")
            return

        if not result.get("ok"):
            self._set_status(self._build_single_export_failure_status(action))
            self._show_export_error(src_name, result, title="出力失敗")
            return

        if action == "current_and_next":
            self._handle_single_export_success_current_and_next(src_str)
            return

        self._handle_single_export_success_current_only(result)

    def _normalize_batch_ui_event(self, event):
        """batch_ui_queue から取り出したイベントを dict に正規化する。"""
        if isinstance(event, tuple) and len(event) == 2:
            event_type, payload = event
            event_type = str(event_type or "")
            if not event_type:
                return None, None
            return event_type, self._normalize_batch_ui_payload_by_type(event_type, dict(payload or {}))
        if isinstance(event, dict):
            payload = dict(event)
            event_type = str(payload.get("type") or "")
            if not event_type:
                return None, None
            return event_type, self._normalize_batch_ui_payload_by_type(event_type, payload)
        return None, None

    def _handle_batch_done_ui_event(self, payload: dict):
        self._set_batch_running_ui(False)
        ok_count = int(payload.get("ok_count", 0))
        ng_count = int(payload.get("ng_count", 0))
        total = int(payload.get("total", 0))
        canceled = bool(payload.get("canceled", False))
        errors = list(payload.get("errors", []))
        skipped = int(payload.get("skipped", 0))

        self._set_status(
            self._build_batch_completion_status(
                ok_count=ok_count, ng_count=ng_count, skipped=skipped, canceled=canceled
            )
        )
        self._show_batch_completion_dialog(
            ok_count=ok_count, ng_count=ng_count, skipped=skipped, total=total, errors=errors, canceled=canceled
        )

    def _handle_worker_fatal_ui_event(self, event_type: str, payload: dict):
        is_single = (str(event_type) == UI_EVENT_SINGLE_FATAL)
        if is_single:
            self._set_single_export_running_ui(False)
            self.single_export_thread = None
            self._reset_single_export_worker_progress()
            self._single_export_request = None
            self._set_status("単一ファイル出力中に致命的エラー")
            title = "単一出力エラー"
            default_message = "単一ファイル出力中に致命的エラーが発生しました。"
        else:
            self._set_batch_running_ui(False)
            self._set_status("一括実行中に致命的エラー")
            title = "一括実行エラー"
            default_message = "一括実行中に致命的エラーが発生しました。"

        user_message = payload.get("user_message") or default_message
        detail = payload.get("detail")
        self._show_error_with_log_hint(title, user_message, detail=detail)

    def _dispatch_batch_ui_event(self, event_type: str, payload: dict):
        """UIスレッド側の batch/single 共通イベント処理。処理したら True。"""
        if event_type in (UI_EVENT_STATUS, UI_EVENT_SINGLE_STATUS):
            self._set_status(payload.get("text", ""))
            return True
        if event_type == UI_EVENT_DONE:
            self._handle_batch_done_ui_event(payload)
            return True
        if event_type == UI_EVENT_SINGLE_DONE:
            self._handle_single_export_done(payload)
            return True
        if event_type in (UI_EVENT_FATAL, UI_EVENT_SINGLE_FATAL):
            self._handle_worker_fatal_ui_event(event_type, payload)
            return True
        return False

    def export_current_only(self):
        """現在のファイルだけ書き出し（非同期）"""
        self._start_single_export_worker("current_only")

    def export_current_and_next(self):
        """
        プレビューしながら1件ずつ処理用：
        現在ファイルを書き出して、次のファイルを開く（非同期）
        """
        self._start_single_export_worker("current_and_next")

    def _set_batch_running_ui(self, running: bool):
        self.batch_running = bool(running)
        if running:
            self.start_btn.state(["disabled"])
            self.cancel_btn.state(["!disabled"])
        else:
            if not self.single_export_running:
                self.start_btn.state(["!disabled"])
            self.cancel_btn.state(["disabled"])

    def cancel_batch_export(self):
        if not self.batch_running:
            return
        self._request_batch_cancel()
        self._set_status("キャンセル要求を受け付けました。現在の処理区切りで停止します…")

    def _post_batch_ui_event(self, event_type: str, **payload):
        event_type = str(event_type or "")
        event = self._normalize_batch_ui_payload_by_type(event_type, payload)
        try:
            self.batch_ui_queue.put(event)
        except (queue.Full, RuntimeError, ValueError) as e:
            self._log_debug("WARN", "batch_ui_queue への通知失敗", exc=e, context={"event_type": event_type})

    def _poll_batch_worker_queue(self):
        self._batch_poll_after_id = None
        if bool(getattr(self, "_app_closing", False)):
            return
        had_events = False
        while True:
            try:
                event = self.batch_ui_queue.get_nowait()
            except queue.Empty:
                break

            event_type, payload = self._normalize_batch_ui_event(event)
            if not event_type or payload is None:
                self._log_debug("WARN", "未知のbatch UIイベント形式", context={"event": repr(event)})
                continue

            had_events = True
            if not self._dispatch_batch_ui_event(event_type, payload):
                self._log_debug("WARN", "未処理のbatch UIイベント", context={"event_type": event_type, "payload": repr(payload)[:400]})

        if self._is_any_worker_running():
            self._schedule_batch_ui_poll(BATCH_POLL_MS)
        elif (not had_events) and (self._batch_poll_after_id is None):
            # 取りこぼし防止（直後に done が入るケース対策）
            batch_alive = bool(self.batch_thread and self.batch_thread.is_alive())
            single_alive = bool(self.single_export_thread and self.single_export_thread.is_alive())
            if batch_alive or single_alive:
                self._schedule_batch_ui_poll(BATCH_POLL_MS)

    def _batch_export_worker_main(self, queue_snapshot, settings_snapshot, export_state_snapshot):
        total = len(queue_snapshot)
        ok_count = 0
        ng_count = 0
        skipped = 0
        errors = []
        canceled = False
        try:
            for i, src in enumerate(queue_snapshot, start=1):
                if self._is_batch_cancel_requested():
                    canceled = True
                    break
                if self._is_generated_crop_file(src, require_source_exists=True):
                    skipped += 1
                    continue

                self._post_batch_ui_event(UI_EVENT_STATUS, text=f"一括実行中: {i}/{total} - {src.name}")
                result = self._export_path(
                    src,
                    cancel_check=self._is_batch_cancel_requested,
                    settings=settings_snapshot,
                    export_state=export_state_snapshot,
                )

                if result.get("canceled"):
                    canceled = True
                    break
                if result.get("ok"):
                    ok_count += 1
                else:
                    ng_count += 1
                    msg = f"{src.name}: {result.get('error', 'unknown error')}"
                    err_detail = dict(result.get("error_detail") or {})
                    if err_detail.get("stage") or (err_detail.get("page") is not None):
                        stage_txt = f" stage={err_detail.get('stage')}" if err_detail.get("stage") else ""
                        page_txt = f" page={err_detail.get('page')}" if (err_detail.get("page") is not None) else ""
                        msg += f" [{(stage_txt + page_txt).strip()}]"
                    if result.get("traceback"):
                        self._log_debug(
                            "ERROR",
                            "一括処理: ファイル失敗",
                            context={"file": str(src), "stage": err_detail.get("stage"), "page": err_detail.get("page")},
                            tb=result["traceback"],
                        )
                    else:
                        self._log_debug(
                            "ERROR",
                            "一括処理: ファイル失敗",
                            context={"file": str(src), "error": result.get("error"), "stage": err_detail.get("stage"), "page": err_detail.get("page")},
                        )
                    errors.append(msg)

            self._post_batch_ui_event(
                UI_EVENT_DONE,
                total=total,
                ok_count=ok_count,
                ng_count=ng_count,
                skipped=skipped,
                errors=errors,
                canceled=canceled,
            )
        except Exception as e:
            # ワーカー境界の最終防御（設計上はここで握ってUIへ通知する）
            tb = self._log_exception("一括ワーカー致命的エラー", exc=e)
            self._post_batch_ui_event(UI_EVENT_FATAL, user_message="一括実行中に内部エラーが発生しました。", detail=str(e), traceback=tb)

    def _start_batch_worker(self):
        if not self.input_queue:
            messagebox.showwarning("未選択", "先にファイルまたはフォルダを選択してください。")
            return
        if self.batch_running:
            messagebox.showinfo("実行中", "すでに一括処理を実行中です。")
            return

        settings_snapshot = dict(self.validate_processing_settings())
        queue_snapshot = [Path(p) for p in self.input_queue]
        export_state_snapshot = self._snapshot_export_state_for_paths(queue_snapshot)
        self._reset_batch_cancel_request()
        self._set_batch_running_ui(True)
        self._set_status("一括実行を開始しました…")
        started = self._start_background_worker(
            target=self._batch_export_worker_main,
            args=(queue_snapshot, settings_snapshot, export_state_snapshot),
            thread_attr="batch_thread",
            worker_label="一括実行",
        )
        if not started:
            self._set_batch_running_ui(False)
            self._set_status("一括実行を開始できませんでした")

    def batch_export_all_no_preview(self):
        """プレビュー無しで全件一括実行（ワーカー版）"""
        self._start_batch_worker()

    # -------------------------
    # 実行モード選択
    # -------------------------
    def start_selected_mode(self):
        """
        - batch: 全件一括実行
        - preview:
            - 単一ファイルなら「実行開始」でそのまま出力
            - 複数ファイルなら「実行開始」で現在を書き出して次へ
        """
        if self._is_any_worker_running():
            if self.batch_running:
                messagebox.showinfo("実行中", "一括処理を実行中です。停止する場合はキャンセルを押してください。")
            else:
                messagebox.showinfo("実行中", "単一ファイル出力を実行中です。完了を待ってください。")
            return

        if not self.input_queue:
            messagebox.showwarning("未選択", "先にファイルまたはフォルダを選択してください。")
            return

        mode = self.exec_mode_var.get()

        # 現在ファイルが未ロードならロード
        if self.current_path is None and self.input_queue:
            self._open_queue_index(self.queue_index)

        if mode == "batch":
            self.batch_export_all_no_preview()
            return

        # previewモードでも「実行開始」で実際に処理する
        if len(self.input_queue) <= 1:
            self.export_current_only()
        else:
            self.export_current_and_next()


    def _cancel_ui_after_callbacks(self):
        for attr in ("_batch_poll_after_id", "_resize_after_id", "_live_preview_param_after_id"):
            after_id = getattr(self, attr, None)
            if after_id is None:
                continue
            try:
                self.root.after_cancel(after_id)
            except TK_UI_HANDLED_EXCEPTIONS:
                pass
            setattr(self, attr, None)

    def _get_close_join_budget_ms(self, attr_name: str) -> int:
        if attr_name == "single_export_thread":
            phase = str(getattr(self, "_single_export_worker_phase", "") or "")
            if phase == SINGLE_EXPORT_PHASE_EXPORTING:
                return int(CLOSE_WORKER_JOIN_TOTAL_SINGLE_EXPORTING_MS)
            return int(CLOSE_WORKER_JOIN_TOTAL_SINGLE_EXPORT_MS)
        if attr_name == "batch_thread":
            return int(CLOSE_WORKER_JOIN_TOTAL_BATCH_MS)
        return int(CLOSE_WORKER_JOIN_TOTAL_DEFAULT_MS)

    def _join_thread_briefly(self, attr_name: str, *, label: str, total_timeout_ms: int = None):
        th = getattr(self, attr_name, None)
        if th is None:
            return False
        try:
            if not th.is_alive():
                return False
        except THREAD_STATE_HANDLED_EXCEPTIONS:
            return False

        total_ms = int(self._get_close_join_budget_ms(attr_name) if total_timeout_ms is None else total_timeout_ms)
        slice_ms = max(1, int(CLOSE_WORKER_JOIN_TIMEOUT_MS))
        attempts = max(1, (max(1, total_ms) + slice_ms - 1) // slice_ms)

        join_attempted = False
        for _ in range(attempts):
            try:
                th.join(slice_ms / 1000.0)
                join_attempted = True
            except RuntimeError as e:
                self._log_debug("WARN", "終了時ワーカーjoin失敗", exc=e, context={"label": label, "slice_ms": slice_ms, "total_timeout_ms": total_ms})
                return True

            still_alive = False
            try:
                still_alive = bool(th.is_alive())
            except THREAD_STATE_HANDLED_EXCEPTIONS:
                still_alive = False

            if not still_alive:
                self._log_debug("INFO", "終了時ワーカー停止を確認", context={"label": label, "waited_up_to_ms": total_ms})
                setattr(self, attr_name, None)
                return True

        self._log_debug(
            "INFO",
            "終了時ワーカーは継続中（daemonのため終了を続行）",
            context={"label": label, "join_slice_ms": slice_ms, "join_total_ms": total_ms},
        )
        return join_attempted

    def _prepare_close_while_workers_running(self):
        if not self._is_any_worker_running():
            return
        worker_labels = self._build_close_worker_labels()
        worker_text = " / ".join(worker_labels) if worker_labels else "実行中処理"
        single_close_note = self._build_single_export_close_note() if ("単一出力" in worker_labels) else ""
        self._set_status(f"終了中: {worker_text} を停止しています…{single_close_note}")
        self._log_debug(
            "INFO",
            "終了要求を受信（実行中ワーカーあり）",
            context={
                "workers": worker_labels,
                "batch_running": bool(self.batch_running),
                "single_export_running": bool(self.single_export_running),
                "single_export_phase": getattr(self, "_single_export_worker_phase", None),
                "single_export_src": getattr(self, "_single_export_worker_src", None),
                "single_export_close_note": single_close_note or None,
            },
        )
        try:
            self._request_batch_cancel()
        except RuntimeError:
            pass
        try:
            self._request_single_export_cancel()
        except RuntimeError:
            pass

    def _is_thread_alive_attr(self, attr_name: str) -> bool:
        th = getattr(self, attr_name, None)
        if th is None:
            return False
        try:
            return bool(th.is_alive())
        except THREAD_STATE_HANDLED_EXCEPTIONS:
            return False

    def _on_close_request(self):
        if bool(getattr(self, "_app_closing", False)):
            return

        if self._is_any_worker_running():
            worker_labels = self._build_close_worker_labels()
            worker_text = " / ".join(worker_labels) if worker_labels else "実行中処理"
            confirm_msg = (
                f"{worker_text} が実行中です。\n\n"
                "停止要求を送って終了しますか？\n"
                "（途中の出力ファイルは未完成になる場合があります）"
            )
            try:
                if not messagebox.askyesno("終了確認", confirm_msg, parent=self.root):
                    self._set_status("終了をキャンセルしました")
                    return
            except TK_UI_HANDLED_EXCEPTIONS as e:
                self._log_debug("WARN", "終了確認ダイアログ表示失敗", exc=e)

        self._app_closing = True

        self._prepare_close_while_workers_running()
        self._cancel_ui_after_callbacks()

        # 実行中ワーカーに停止要求を送った後、フェーズ別の待機時間で順に待つ。未停止なら強制終了確認を出す。
        if self._is_thread_alive_attr("batch_thread"):
            self._set_status("終了中: 一括処理の停止を待機しています…")
        self._join_thread_briefly("batch_thread", label="batch", total_timeout_ms=self._get_close_join_budget_ms("batch_thread"))

        if self._is_thread_alive_attr("single_export_thread"):
            self._set_status(f"終了中: 単一出力の停止を待機しています…{self._build_single_export_close_note()}")
        self._join_thread_briefly("single_export_thread", label="single_export", total_timeout_ms=self._get_close_join_budget_ms("single_export_thread"))

        if self._is_thread_alive_attr("batch_thread") or self._is_thread_alive_attr("single_export_thread"):
            force_close = True
            worker_labels = self._build_close_worker_labels()
            worker_text = " / ".join(worker_labels) if worker_labels else "実行中処理"
            force_msg = (
                f"{worker_text} がまだ停止していません。\n\n"
                "このまま終了を強行しますか？\n"
                "（出力途中のファイルが残る可能性があります）"
            )
            try:
                force_close = bool(messagebox.askyesno("強制終了の確認", force_msg, parent=self.root))
            except TK_UI_HANDLED_EXCEPTIONS as e:
                self._log_debug("WARN", "強制終了確認ダイアログ表示失敗", exc=e)
            if not force_close:
                self._app_closing = False
                self._set_status("終了をキャンセルしました（処理継続中）")
                self._ensure_batch_ui_polling()
                return

        try:
            self._reset_single_export_worker_progress()
        except (AttributeError, RuntimeError):
            pass

        try:
            self.close_current_doc()
        except RuntimeError as e:
            self._log_debug("WARN", "終了時のPDF close失敗", exc=e)
        try:
            self._safe_close_image(self.preview_pil, label="app_close", context={"path": str(self.current_path) if self.current_path else None})
            self.preview_pil = None
        except RuntimeError:
            pass
        try:
            self.root.destroy()
        except TK_UI_HANDLED_EXCEPTIONS:
            pass

    def run(self):
        self.root.mainloop()


# Tkルート生成（DnD有無で切替）
class CropApp(CropAppBase):
    def __init__(self):
        if DND_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        super().__init__()


if __name__ == "__main__":
    app = CropApp()
    if _STARTUP_OPTIONAL_MISSING:
        optional_lines = []
        for disp, pkg, feature in _STARTUP_OPTIONAL_MISSING:
            optional_lines.append(f"- {disp}（{feature} / pip install {pkg}）")
        _show_startup_library_alert(
            "一部機能が利用できません（ライブラリ不足）",
            "不足しているライブラリがあります。アプリは起動しますが、一部機能が使えません。\n\n"
            + "\n".join(optional_lines),
            kind="warning",
            parent=app.root,
        )
    app.run()
