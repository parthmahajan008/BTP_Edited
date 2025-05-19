#!/bin/bash

DATASET=PopQA

# Create directories if they don't exist
mkdir -p dataset/${DATASET}/with_rationale

# Execute the inference script with Azure OpenAI
python src/inference.py \
  --dataset_name $DATASET \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --n_docs 5 \
  --output_dir dataset/${DATASET} \
  --do_rationale_generation \
  --use_azure_openai \
  --azure_api_key $AZURE_API_KEY \
  --azure_endpoint_url $AZURE_ENDPOINT_URL \
  --azure_api_version $AZURE_API_VERSION \
  --azure_deployment_name $AZURE_DEPLOYMENT_NAME \
  --max_concurrent_requests 200 