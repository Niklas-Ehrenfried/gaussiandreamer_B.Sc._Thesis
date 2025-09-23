#!/bin/bash

#SBATCH --job-name=ThreeStudio_Multi
#SBATCH --cpus-per-task=4
#SBATCH --partition=week
#SBATCH --mem-per-cpu=40G
#SBATCH --gres=gpu:A4000:1
#SBATCH --time=7-00:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=niklas.ehrenfried@student.uni-tuebingen.de


# Define color codes for pretty output
CSI="\033["
RESET="${CSI}0m"
BOLD="${CSI}1m"
BLUE="${CSI}34m"
CYAN="${CSI}36m"
GREEN="${CSI}32m"
YELLOW="${CSI}33m"
MAGENTA="${CSI}35m"
RED="${CSI}31m"

set -u
shopt -s nullglob

DRY_RUN=0
echo "[DEBUG] Starting batch script, DRY_RUN=${DRY_RUN}"

BRANCH="master"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --batch-dir)
            BATCH_DIR="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        *)
            echo "[DEBUG] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "${BATCH_DIR:-}" || -z "${BRANCH:-}" ]]; then
    echo "[DEBUG] Usage: $0 --batch-dir <dir> --branch <branch>" >&2
    exit 1
fi

# Validate directories
[[ -d "${BATCH_DIR}" ]] || { echo "[DEBUG] Batch directory does not exist: ${BATCH_DIR}" >&2; exit 1; }


echo "[DEBUG] Using batch directory: ${BATCH_DIR}"
echo "[DEBUG] Using Git branch: ${BRANCH}"
echo "[DEBUG] Using batch directory: ${BATCH_DIR}"

[[ -d "${BATCH_DIR}" ]] || { echo "[DEBUG] Batch directory missing" >&2; exit 1; }
[[ -d "${BATCH_DIR}/ply" ]] || { echo "[DEBUG] ply subdirectory missing" >&2; exit 1; }
[[ -d "${BATCH_DIR}/txt" ]] || { echo "[DEBUG] txt subdirectory missing" >&2; exit 1; }
[[ -d "${BATCH_DIR}/configs" ]] || { echo "[DEBUG] configs subdirectory missing" >&2; exit 1; }

SCRATCH_DIR="/scratch/${SLURM_JOB_ID}"
PERMANENT_OUTPUT_DIR="${HOME}/ResultsInit"
mkdir -p "${SCRATCH_DIR}" || exit 1
mkdir -p "${PERMANENT_OUTPUT_DIR}" || exit 1
echo "[DEBUG] Scratch dir: ${SCRATCH_DIR}, Permanent output: ${PERMANENT_OUTPUT_DIR}"

SCRATCH_BATCH="${SCRATCH_DIR}/batch"
cp -a "${BATCH_DIR}" "${SCRATCH_BATCH}" || { echo "[DEBUG] Failed to copy batch to scratch" >&2; exit 1; }
CONFIG_DIR="${SCRATCH_BATCH}/configs"
echo "[DEBUG] Copied batch to scratch, config dir: ${CONFIG_DIR}"



printf "\n\n\n${BOLD}${CYAN}=================================================${RESET}\n"
printf "${BOLD}${CYAN}=== Starting ThreeStudio Gaussian Dreamer Job ===${RESET}\n"
printf "${BOLD}${CYAN}=================================================${RESET}\n\n"

# Job details
printf "${BOLD}${BLUE}--- JOB DETAILS${RESET}\n"
printf "  ${GREEN}Job ID:${RESET}               ${SLURM_JOB_ID}\n"
printf "  ${GREEN}Host:${RESET}                 $(hostname)\n"
printf "  ${GREEN}Working Dir:${RESET}          $(pwd)\n\n"

# GPU information
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
CUDA_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1)
printf "${BOLD}${BLUE}--- GPU INFORMATION${RESET}\n"
printf "  ${YELLOW}GPU Detected:${RESET}        ${GPU_NAME}\n"
printf "  ${YELLOW}CUDA Driver Ver:${RESET}     ${CUDA_VERSION}\n"
printf "  ${YELLOW}Compute Capability:${RESET}  ${CUDA_CAPABILITY}\n\n\n\n"



printf "${BOLD}${BLUE}--- Cloning ThreeStudio Gaussian Dreamer${RESET}\n"
if [[ -f "$HOME/.git_pat" ]]; then
  GITHUB_TOKEN=$(cat "$HOME/.git_pat")
else
  echo "[DEBUG] Git PAT not found" >&2
  exit 1
fi

mkdir -p "${SCRATCH_DIR}/custom"
touch "${SCRATCH_DIR}/custom/__init__.py"
git clone -q -b "${BRANCH}" "https://Niklas-Ehrenfried:${GITHUB_TOKEN}@github.com/Niklas-Ehrenfried/gaussiandreamer_B.Sc._Thesis.git" ${SCRATCH_DIR}/custom/threestudio_gaussiandreamer

echo "[DEBUG] Cloned custom repo"
printf "Latest commit: $(git -C "${SCRATCH_DIR}/custom/threestudio_gaussiandreamer" log -1 --pretty=format:'%h %s')\n"


printf "${BOLD}${BLUE}--- Creating symlinks for custom configs${RESET}\n"
for mod in threestudio_lrm threestudio_shap_e shap_e threestudio_mvdream; do
  ln -sf "/opt/threestudio/custom/$mod" "${SCRATCH_DIR}/custom/$mod" || { printf "${RED}Failed to link $mod${RESET}\n"; exit 1; }
  printf "Linked ${mod}\n"
done
ln -sf "/opt/threestudio/custom/threestudio_gaussiandreamer/diff-gaussian-rasterization" "${SCRATCH_DIR}/custom/threestudio_gaussiandreamer/diff-gaussian-rasterization"
printf "Linked rasterization module\n"
ln -sf "/opt/threestudio/custom/threestudio_gaussiandreamer/simple-knn" "${SCRATCH_DIR}/custom/threestudio_gaussiandreamer/simple-knn"
printf "Linked knn module\n\n"


mkdir ${SCRATCH_DIR}/shap_e_cache || { echo "ERROR: Failed to create shap-e cache directory!\n"; exit 1; }
printf "Cache directory: ${SCRATCH_DIR}/shap_e_cache\n\n\n"

CONFIG_PATH="${SCRATCH_DIR}/custom/threestudio_gaussiandreamer/configs/gaussiandreamer_mvdream.yaml"
echo "[DEBUG] Default CONFIG_PATH: ${CONFIG_PATH}"

config_files=( "${CONFIG_DIR}"/*.txt )
if [[ ${#config_files[@]} -eq 0 ]]; then
  echo "[DEBUG] No config files found in ${CONFIG_DIR}" >&2
  exit 1
fi
echo "[DEBUG] Found ${#config_files[@]} config files"

for CONFIG_FILE in "${config_files[@]}"; do
    CONFIG_NAME=$(basename "${CONFIG_FILE}" .txt)
    printf "${BOLD}${CYAN}============================================================${RESET}\n"
    printf "${BOLD}${CYAN}=== Starting ThreeStudio Gaussian Dreamer Training Phase ===${RESET}\n"
    printf "${BOLD}${CYAN}============================================================${RESET}\n"
    printf "Running config: \"${CONFIG_NAME}\"\n\n"

    ITERATIONS=3000
    INIT_PROMPT="a cube"

    LOSS_LAMBDA_SDS=1
    LOSS_LAMBDA_MASKED_SDS=0.0
    LOSS_LAMBDA_ALPHA_MASKED_SDS=0.3
    LOSS_LAMBDA_DEPTH_MASKED_SDS=0.7

    LOSS_LAMBDA_ALPHA_SILHOUETTE_LOSS=0
    LOSS_LAMBDA_DEPTH_SILHOUETTE_LOSS=0

    LOSS_LAMBDA_SCALE_OUTLIER=0.0

    LOSS_MASK_FILL_METHOD="distance"
    LOSS_MASK_DISTANCE_MAX_RADIUS=24
    LOSS_DISTANCE_SIGMA=16.0
    LOSS_LAMBDA_MASK_PENETRATION=0

    LOSS_CHAMFER=0

    GEOM_MULTIPLIER_LR=0.01
    GEOM_POSITION_LR=0.00005
    GEOM_SCALE_LR=0.005
    GEOM_FEATURE_LR=0.01
    GEOM_OPACITY_LR=0.01
    GEOM_ROTATION_LR=0.001

    LOSS_LAMBDA_SPARSITY=0.0
    LOSS_LAMBDA_POSITION=0.0
    LOSS_LAMBDA_OPACITY=0.0
    LOSS_LAMBDA_SCALES=0.0
    LOSS_LAMBDA_TV_LOSS=0.0
    LOSS_LAMBDA_DEPTH_TV_LOSS=0.0

    GEOM_WARMUP_COLOR_ITERS=1000
    GEOM_RAMP_UP_STRUCTURE_ITERS=1000
    GEOM_PRUNING_START=2800
    GEOM_DENSIFY_FROM_ITER=500
    GEOM_DENSIFY_UNTIL_ITER=1500
    GEOM_DENSIFICATION_INTERVAL=200
    GEOM_PRUNE_FROM_ITER=500
    GEOM_PRUNE_UNTIL_ITER=1500
    GEOM_PRUNE_INTERVAL=200

    source "${CONFIG_FILE}"

    ply_list=( "${SCRATCH_BATCH}/ply"/*.ply )
    if [[ ${#ply_list[@]} -eq 0 ]]; then
        echo "[DEBUG] No PLY files found in ${SCRATCH_BATCH}/ply, skipping config ${CONFIG_NAME}" >&2
        continue
    fi
    echo "[DEBUG] Found ${#ply_list[@]} PLY files for config ${CONFIG_NAME}"

    for ply_file in "${ply_list[@]}"; do
        base_name=$(basename "${ply_file}" .ply)
        printf "${BOLD}${CYAN}===========${RESET}\n"
        printf "${BOLD}${CYAN}=== Job ===${RESET}\n"
        printf "${BOLD}${CYAN}===========${RESET}\n"
        printf "Running file: \"${base_name}\"\n\n"
        # per-run working dir in scratch (no need to copy the ply because we're using the copy already)
        item_scratch_dir="${SCRATCH_DIR}/${CONFIG_NAME}_${base_name}"
        mkdir -p "${item_scratch_dir}" || { echo "mkdir failed for ${item_scratch_dir}" >&2; continue; }

        # read the prompt exactly as-is (preserve leading/trailing whitespace and any quotes)
        txt_file="${SCRATCH_BATCH}/txt/${base_name}.txt"
        [[ -f "${txt_file}" ]] || { echo "missing prompt ${txt_file}, skipping ${base_name}" >&2; rm -rf "${item_scratch_dir}"; continue; }

        IFS= read -r PROMPT_CONTENT < "${txt_file}"


        INIT_ARGS=(
            system.geometry.geometry_convert_from="${ply_file}"
            system.geometry.load_ply_only_vertex=true
            system.geometry.sh_degree=0
            system.geometry.pc_init_radius=1.0
            # system.geometry.warmup_color_iters=${GEOM_WARMUP_COLOR_ITERS}
            # system.geometry.ramp_up_structure_iters=${GEOM_RAMP_UP_STRUCTURE_ITERS}
            system.geometry.pruning_start=${GEOM_PRUNING_START}
            system.geometry.densify_from_iter=${GEOM_DENSIFY_FROM_ITER}
            system.geometry.densify_until_iter=${GEOM_DENSIFY_UNTIL_ITER}
            system.geometry.densification_interval=${GEOM_DENSIFICATION_INTERVAL}
            system.geometry.prune_from_iter=${GEOM_PRUNE_FROM_ITER}
            system.geometry.prune_until_iter=${GEOM_PRUNE_UNTIL_ITER}
            system.geometry.prune_interval=${GEOM_PRUNE_INTERVAL}

            # system.geometry.multiplier_lr=${GEOM_MULTIPLIER_LR}
            # system.geometry.position_lr=${GEOM_POSITION_LR}
            # system.geometry.scale_lr=${GEOM_SCALE_LR}
            # system.geometry.feature_lr=${GEOM_FEATURE_LR}
            # system.geometry.opacity_lr=${GEOM_OPACITY_LR}
            # system.geometry.rotation_lr=${GEOM_ROTATION_LR}
            
            # system.loss.lambda_alpha_silhouette_loss=${LOSS_LAMBDA_ALPHA_SILHOUETTE_LOSS}
            # system.loss.lambda_depth_silhouette_loss=${LOSS_LAMBDA_DEPTH_SILHOUETTE_LOSS}
            # system.loss.lambda_masked_sds=${LOSS_LAMBDA_MASKED_SDS}
            # system.loss.lambda_alpha_masked_sds=${LOSS_LAMBDA_ALPHA_MASKED_SDS}
            # system.loss.lambda_depth_masked_sds=${LOSS_LAMBDA_DEPTH_MASKED_SDS}
            system.loss.lambda_sds=${LOSS_LAMBDA_SDS}
            # system.loss.lambda_scale_outlier=${LOSS_LAMBDA_SCALE_OUTLIER}
            # system.loss.mask_fill_method=${LOSS_MASK_FILL_METHOD}
            # system.loss.mask_distance_max_radius=${LOSS_MASK_DISTANCE_MAX_RADIUS}
            # system.loss.distance_sigma=${LOSS_DISTANCE_SIGMA}
            # system.loss.lambda_mask_penetration=${LOSS_LAMBDA_MASK_PENETRATION}
            system.loss.lambda_chamfer=${LOSS_CHAMFER}
            system.loss.lambda_sparsity=${LOSS_LAMBDA_SPARSITY}
            system.loss.lambda_position=${LOSS_LAMBDA_POSITION}
            system.loss.lambda_opacity=${LOSS_LAMBDA_OPACITY}
            system.loss.lambda_scales=${LOSS_LAMBDA_SCALES}
            system.loss.lambda_tv_loss=${LOSS_LAMBDA_TV_LOSS}
            system.loss.lambda_depth_tv_loss=${LOSS_LAMBDA_DEPTH_TV_LOSS}

        )

        # run singularity for this item — no per-item log redirection (sbatch will collect stdout/stderr)
        # singularity run --nv --pwd "${item_scratch_dir}" ThreeStudio.sif \
        #     --config "${CONFIG_PATH}" \
        #     trainer.max_steps="${ITERATIONS}" --train --gpu 0 \
        #     system.prompt_processor.prompt="${PROMPT_CONTENT}" \
        #     "${INIT_ARGS[@]}" exp_root_dir="${item_scratch_dir}"
        # TRAIN_EXIT=$?

        CMD=(singularity run --nv --pwd "${SCRATCH_DIR}" ThreeStudio.sif \
        --config "${CONFIG_PATH}" \
        trainer.max_steps="${ITERATIONS}" --train --gpu 0 \
        system.prompt_processor.prompt="${PROMPT_CONTENT}" \
        "${INIT_ARGS[@]}" exp_root_dir="${item_scratch_dir}")

        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "[DRY RUN] Would execute singularity run for ${base_name}"
            echo "${CMD[@]}"
            TRAIN_EXIT=0
        else
            "${CMD[@]}"
            TRAIN_EXIT=$?
        fi

        if [[ ${TRAIN_EXIT} -ne 0 ]]; then
            echo "Singularity run failed for ${base_name}, exit=${TRAIN_EXIT}. Check sbatch log for details." >&2
            # keep item_scratch_dir for debugging; continue to next ply
            continue
        fi

        if [[ "${DRY_RUN}" -eq 1 ]]; then
            # simulate a run output directory for testing copy logic
            mkdir -p "${item_scratch_dir}/gs-sds-mvdream/${base_name}@001"
            touch "${item_scratch_dir}/gs-sds-mvdream/${base_name}@001/${base_name}.txt"
        fi

        # find produced run directory
        RUN_DIR="$(ls -dt "${item_scratch_dir}"/gs-sds-mvdream/*@* 2>/dev/null | head -n 1 || true)"
        if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
            echo "Could not find run output for ${base_name}, check ${item_scratch_dir}." >&2
            continue
        fi

        # prune large/unnecessary subpaths
        rm -rf "${RUN_DIR}/ckpts" 2>/dev/null || true
        rm -rf "${RUN_DIR}/code" 2>/dev/null || true
        rm -rf "${RUN_DIR}/tb_logs" 2>/dev/null || true
        rm -rf "${RUN_DIR}/save/point_cloud.ply" 2>/dev/null || true

        # determine unique destination name under home
        DEST_DIR_BASE="${PERMANENT_OUTPUT_DIR}/gs-sds-mvdream/${CONFIG_NAME}/$(basename "${RUN_DIR}")"
        DEST_DIR="${DEST_DIR_BASE}"
        i=1
        while [[ -e "${DEST_DIR}" ]]; do
            DEST_DIR="${DEST_DIR_BASE}(${i})"
            ((i++))
        done
        mkdir -p "$(dirname "${DEST_DIR}")" || true

        mkdir -p "${DEST_DIR}"
        cp -R "${RUN_DIR}/." "${DEST_DIR}/" || {
            echo "Failed to copy result to permanent output dir ${DEST_DIR}, check scratch ${RUN_DIR}." >&2
            mkdir -p "${HOME}/FaildCP"
            cp -R "${RUN_DIR}/." "${HOME}/FaildCP" || echo "Also failed to copy to ${HOME}/FaildCP, scratch data will be lost." >&2
            continue
        }
        # clean per-run scratch to keep scratch tidy
        rm -rf "${item_scratch_dir}" 2>/dev/null || true
    done
done

echo "[DEBUG] Batch script finished"
exit 0
