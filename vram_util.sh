#!/bin/bash

THRESHOLD=28000  # VRAM 임계값(MiB)
SLEEP_INTERVAL=10 # 반복 주기 (초 단위, '=' 기준으로 띄어쓰기 하지 마세요(e.g., SLEEP_INTERVAL = 3600 (X)). 값이 SLEEP_INTERVAL에 할당 안 돼요. 60=1분, 3600=1시간, 86400=1일, 604800=1주, 이건 계산에 참고하세요~)
EXCLUDED_GPUS=(0)  # 제외할 GPU 번호들 (예: GPU 0과 1을 제외하려면 (0 1), 모든 GPU 사용하려면 () 빈 배열로 설정)

while true; do
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
    SELECTED_GPUS=()

    for ((i=0; i<GPU_COUNT; i++)); do
        # 제외된 GPU인지 확인
        if [[ " ${EXCLUDED_GPUS[@]} " =~ " ${i} " ]]; then
            echo "$(date "+%Y-%m-%d %H:%M:%S") GPU $i: Excluded from selection"
            continue
        fi

        # VRAM 정보 가져오기
        Free_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n "$((i+1))p" | tr -d '[:space:]')
        
        # GPU 사용률 확인
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sed -n "$((i+1))p" | tr -d '[:space:]')

        # GPU에서 실행 중인 프로세스 개수 확인
        GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | sed -n "$((i+1))p" | tr -d '[:space:]')
        PROCESS_COUNT=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$GPU_UUID" 2>/dev/null)

        echo "$(date "+%Y-%m-%d %H:%M:%S") GPU $i: Free VRAM ${Free_VRAM}MiB, GPU Util: ${GPU_UTIL}%, Processes: ${PROCESS_COUNT}"
        
        # VRAM 임계값 만족하고 실행 중인 프로세스가 0개 이상인 경우만 선택
        # argument: -le, -ge, -eq, -ne, -lt, -gt, -le, -ge, -eq, -ne, -lt, -gt
        if [ "$Free_VRAM" -ge "$THRESHOLD" ] && [ "$PROCESS_COUNT" -ge 0 ]; then
            SELECTED_GPUS+=($i) # threshold 이상이고 프로세스가 없는 경우 사용할 gpu 번호 저장
            echo "$(date "+%Y-%m-%d %H:%M:%S") GPU $i: Available for use"
        elif [ "$Free_VRAM" -ge "$THRESHOLD" ] && [ "$PROCESS_COUNT" -gt 0 ]; then
            echo "$(date "+%Y-%m-%d %H:%M:%S") GPU $i: Skipped due to running ${PROCESS_COUNT} process(es)"
        elif [ "$Free_VRAM" -lt "$THRESHOLD" ]; then
            echo "$(date "+%Y-%m-%d %H:%M:%S") GPU $i: Insufficient VRAM (${Free_VRAM}MiB < ${THRESHOLD}MiB)"
        fi
    done

    if [ ${#SELECTED_GPUS[@]} -gt 0 ]; then
		GPU_LIST=$(IFS=,; echo "${SELECTED_GPUS[*]}")
		echo "[$(date "+%Y-%m-%d %H:%M:%S")] Using GPUs: $GPU_LIST"
		# default: CUDA_VISIBLE_DEVICES=$GPU_LIST python <your_python_file> <your_arguments>
		CMD="CUDA_VISIBLE_DEVICES=$GPU_LIST python GUI-Actor/eval/screenSpot_pro.py"
		echo "[$(date "+%Y-%m-%d %H:%M:%S")] Running: $CMD"
		eval "$CMD"
	else
		echo "[$(date "+%Y-%m-%d %H:%M:%S")] No available GPUs found. Waiting..."
	fi

    echo "Sleeping for $SLEEP_INTERVAL seconds..."
    sleep $SLEEP_INTERVAL
done
