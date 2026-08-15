#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <patient_id>"
  echo "Example: $0 1"
  exit 1
fi

PATIENT_ID="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
PATIENT_DIR="$ROOT_DIR/data/$PATIENT_ID"
MODEL_PATH="$PATIENT_DIR/$PATIENT_ID"
MASKS_DIR="$PATIENT_DIR/masks"
LUNG_MASKS_DIR="$PATIENT_DIR/lung_masks"
ORIGINALS_DIR="$PATIENT_DIR/original"

ATTACKED_PLY="$MODEL_PATH/attacked_clf.ply"
ATTACKED_MODEL_PATH="$PATIENT_DIR/${PATIENT_ID}_attacked_clf"
ATTACKED_RENDER_DIR="$ATTACKED_MODEL_PATH/render_img"
CLEAN_RENDER_DIR="$MODEL_PATH/render_img"

ANALYSIS_DIR="$ROOT_DIR/analysis/run${PATIENT_ID}"
DIFF_DIR="$ANALYSIS_DIR/diffs"
COMP_DIR="$ANALYSIS_DIR/components"
ANNOTATED_DIR="$ANALYSIS_DIR/annotated"
MASK_CHECK_CSV="$ANALYSIS_DIR/mask_tumor_check.csv"

# Attack defaults (override any of these by exporting them before running)
SIGMA="${SIGMA:-0.5}"
REMOVE_TOP_PCT="${REMOVE_TOP_PCT:-10.0}"
PROPS="${PROPS:-f_dc_0 f_dc_1 f_dc_2}"
EPS="${EPS:-0.5}"
STEPS="${STEPS:-10}"
ATTACK_MODE="${ATTACK_MODE:-untargeted}"
TARGET_YEAR="${TARGET_YEAR:-5}"
MAX_SLICES="${MAX_SLICES:-50}"
CLASSIFIER_DEVICE="${CLASSIFIER_DEVICE:-cuda}"
CLF_SPATIAL_SIZE="${CLF_SPATIAL_SIZE:-128 128}"
N_ENSEMBLE_LOAD="${N_ENSEMBLE_LOAD:-0}"
N_ENSEMBLE_MODELS="${N_ENSEMBLE_MODELS:-0}"

# Analysis defaults
MIN_AREA="${MIN_AREA:-20}"
CLOSING_RADIUS="${CLOSING_RADIUS:-5}"
RADIUS_SCALE="${RADIUS_SCALE:-1.5}"
MIN_RADIUS="${MIN_RADIUS:-10}"
THICKNESS="${THICKNESS:-2}"

# Validate expected input folders
[[ -d "$MODEL_PATH" ]] || { echo "Missing model path: $MODEL_PATH"; exit 1; }
[[ -d "$MASKS_DIR" ]] || { echo "Missing masks dir: $MASKS_DIR"; exit 1; }
[[ -d "$LUNG_MASKS_DIR" ]] || { echo "Missing lung masks dir: $LUNG_MASKS_DIR"; exit 1; }
[[ -d "$ORIGINALS_DIR" ]] || { echo "Missing originals dir: $ORIGINALS_DIR"; exit 1; }

# Runtime env setup
export MEDGS_ROOT="${MEDGS_ROOT:-$ROOT_DIR/submodules/MedGS}"
export PYTHONPATH="$ROOT_DIR/submodules/Sybil/src:$ROOT_DIR/submodules/Sybil/src_sybil:$ROOT_DIR/submodules/MedGS:${PYTHONPATH:-}"

CLASSIFIER_CONFIG_DIR="${CLASSIFIER_CONFIG_DIR:-$ROOT_DIR/submodules/Sybil/configs}"
CLASSIFIER_CONFIG_NAME="${CLASSIFIER_CONFIG_NAME:-nlst_sybil_ensemble_inference}"

# Auto-detect latest MedGS iteration for render copy path.
LATEST_ITER="$(ls -1 "$MODEL_PATH/point_cloud" | sed -n 's/^iteration_//p' | sort -n | tail -1)"
if [[ -z "$LATEST_ITER" ]]; then
  echo "Could not detect iteration under $MODEL_PATH/point_cloud"
  exit 1
fi

echo "============================================================"
echo "Pipeline start for patient: $PATIENT_ID"
echo "Model path:  $MODEL_PATH"
echo "Iteration:   $LATEST_ITER"
echo "Output root: $ANALYSIS_DIR"
echo "============================================================"

mkdir -p "$ANALYSIS_DIR"

# 1) Attack

echo "\n[1/6] Running attack"
python "$ROOT_DIR/main.py" \
  --model_path "$MODEL_PATH" \
  --source_path "$PATIENT_DIR" \
  --masks "$MASKS_DIR" \
  --output_ply "$ATTACKED_PLY" \
  --sigma "$SIGMA" --remove_top_pct "$REMOVE_TOP_PCT" \
  --props $PROPS \
  --eps "$EPS" --steps "$STEPS" \
  --attack_mode "$ATTACK_MODE" \
  --classifier_config_dir "$CLASSIFIER_CONFIG_DIR" \
  --classifier_config_name "$CLASSIFIER_CONFIG_NAME" \
  --target_year "$TARGET_YEAR" \
  --max_slices "$MAX_SLICES" \
  --clf_spatial_size $CLF_SPATIAL_SIZE \
  --n_ensemble_load "$N_ENSEMBLE_LOAD" \
  --n_ensemble_models "$N_ENSEMBLE_MODELS" \
  --classifier_device "$CLASSIFIER_DEVICE" \

# 2) Render attacked copy
# Rebuild attacked model folder so render.py reads attacked PLY as point_cloud.
echo "\n[2/6] Rendering attacked model"
rm -rf "$ATTACKED_MODEL_PATH"
cp -r "$MODEL_PATH" "$ATTACKED_MODEL_PATH"
cp "$ATTACKED_PLY" "$ATTACKED_MODEL_PATH/point_cloud/iteration_${LATEST_ITER}/point_cloud.ply"

python "$ROOT_DIR/submodules/MedGS/render.py" \
  --model_path "$ATTACKED_MODEL_PATH" \
  --source_path "$PATIENT_DIR" \
  --iteration "$LATEST_ITER" \
  --camera mirror \
  --pipeline img

# 3) Diffs
echo "\n[3/6] Generating attacked-vs-clean diffs"
python -c "from analysis.image_diff import generate_image_diffs as f; f('$ATTACKED_RENDER_DIR','$CLEAN_RENDER_DIR','$DIFF_DIR')"

# 4) Connected components in lungs
echo "\n[4/6] Running connected-components analysis"
python -c "from analysis.component_analysis import analyze_connected_components as f; f(diff_folder='$DIFF_DIR', output_folder='$COMP_DIR', lung_masks_folder='$LUNG_MASKS_DIR', threshold=None, min_area=$MIN_AREA, closing_radius=$CLOSING_RADIUS)"

# 5) Overlay circles
echo "\n[5/6] Drawing centroid overlays"
python -c "from analysis.overlay_circles import draw_circles_on_originals as f; f(csv_path='$COMP_DIR/centroids.csv', originals_folder='$ORIGINALS_DIR', output_folder='$ANNOTATED_DIR', radius_mode='area', scale=$RADIUS_SCALE, min_radius=$MIN_RADIUS, thickness=$THICKNESS, color=(255,0,0))"

# 6) Simple mask white-pixel check
echo "\n[6/6] Checking tumor-positive mask slices"
python "$ROOT_DIR/analysis/check_tumor_masks.py" \
  --masks_dir "$MASKS_DIR" \
  --threshold 128 \
  --min_white_pixels 1 \
  --output_csv "$MASK_CHECK_CSV"

echo "\nDone."
echo "Attacked PLY:      $ATTACKED_PLY"
echo "Diffs:             $DIFF_DIR"
echo "Components:        $COMP_DIR"
echo "Annotated images:  $ANNOTATED_DIR"
echo "Mask check CSV:    $MASK_CHECK_CSV"
