#!/usr/bin/bash

if [[ -z "${NUM_WORKERS}" ]]; then
  NUM_WORKERS=1
fi

echo NUM_WORKERS set to $NUM_WORKERS


faststream run --workers $NUM_WORKERS infobip_service.kafka_downloading:app &> ./downloading.log & 

# fastkafka run --num-workers $NUM_WORKERS --kafka-broker $KAFKA_BROKER infobip_kafka_service.training:app &> ./training.log & 

# venv/bin/python -m infobip_kafka_service.scheduler &> ./scheduler.log & 

# tail -f training.log

tail -f downloading.log
