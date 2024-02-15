#!/usr/bin/bash

if [[ -z "${NUM_WORKERS}" ]]; then
  NUM_WORKERS=1
fi

echo NUM_WORKERS set to $NUM_WORKERS


faststream run --workers $NUM_WORKERS infobip_service.kafka_downloading:app &> ./downloading.log &

faststream run --workers $NUM_WORKERS infobip_service.kafka_training:app &> ./training.log &

python3 -m infobip_service.scheduler &> ./scheduler.log &

# tail -f downloading.log

tail -f training.log
