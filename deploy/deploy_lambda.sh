#!/usr/bin/env bash
# Deploy the prediction API to AWS Lambda (container image + public function URL).
# Safe to re-run: existing resources are reused and the function is updated to the new image.
#
#   bash deploy/deploy_lambda.sh
#
# Needs: AWS CLI v2 (recent), Docker running, models/lgbm_improved.txt, and AWS credentials.
set -euo pipefail
cd "$(dirname "$0")/.."
source deploy/config.sh

TAG="$(git rev-parse --short HEAD)"
IMAGE="${ECR}/${REPO}:${TAG}"
MODEL_KEY="models/lgbm_improved-${TAG}.txt"
step() { printf "\n==> %s\n" "$1"; }

[ -f models/lgbm_improved.txt ] || { echo "models/lgbm_improved.txt not found - run: python -m src.pipeline"; exit 1; }

step "1/7 S3: upload the model to s3://${BUCKET}/${MODEL_KEY}"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3 mb "s3://${BUCKET}" --region "$REGION"
fi
aws s3 cp models/lgbm_improved.txt "s3://${BUCKET}/${MODEL_KEY}" --region "$REGION"

step "2/7 ECR: create the image repository (if needed) and log Docker in"
if ! aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "$REPO" --region "$REGION" \
    --image-scanning-configuration scanOnPush=true >/dev/null
fi
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR"

step "3/7 Docker: build for arm64 (AWS Graviton) and push ${IMAGE}"
# --provenance=false: Lambda only accepts a plain single-platform image manifest
docker buildx build --platform linux/arm64 --provenance=false \
  --build-arg PYTHON_VERSION="$(cat .python-version)" -t "$IMAGE" --push .

step "4/7 IAM: role that lets the function write logs and read the model"
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" \
    --assume-role-policy-document file://deploy/lambda-trust-policy.json >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "Waiting 15 s for the new role to become usable..."
  sleep 15
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name read-model --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{\"Effect\": \"Allow\", \"Action\": \"s3:GetObject\",
                   \"Resource\": \"arn:aws:s3:::${BUCKET}/models/*\"}]
}"
ROLE_ARN="$(aws iam get-role --role-name "$ROLE" --query Role.Arn --output text)"

step "5/7 Lambda: create or update the function"
ENV_VARS="Variables={MODEL_S3_URI=s3://${BUCKET}/${MODEL_KEY},MODEL_VERSION=${TAG}}"
if aws lambda get-function --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "$FUNCTION" --image-uri "$IMAGE" \
    --region "$REGION" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
  aws lambda update-function-configuration --function-name "$FUNCTION" --environment "$ENV_VARS" \
    --region "$REGION" >/dev/null
  aws lambda wait function-updated-v2 --function-name "$FUNCTION" --region "$REGION"
else
  aws lambda create-function --function-name "$FUNCTION" --region "$REGION" \
    --package-type Image --code ImageUri="$IMAGE" --role "$ROLE_ARN" \
    --architectures arm64 --memory-size 2048 --timeout 30 \
    --environment "$ENV_VARS" >/dev/null
  aws lambda wait function-active-v2 --function-name "$FUNCTION" --region "$REGION"
fi

step "6/7 Public HTTPS URL (function URL, no login needed)"
if ! aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" >/dev/null 2>&1; then
  aws lambda create-function-url-config --function-name "$FUNCTION" --auth-type NONE \
    --region "$REGION" >/dev/null
  # Since October 2025 a public function URL needs BOTH of these permissions
  aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" \
    --statement-id FunctionUrlPublicAccess --action lambda:InvokeFunctionUrl \
    --principal "*" --function-url-auth-type NONE >/dev/null
  aws lambda add-permission --function-name "$FUNCTION" --region "$REGION" \
    --statement-id FunctionUrlInvokeFunction --action lambda:InvokeFunction \
    --principal "*" --invoked-via-function-url >/dev/null
fi
URL="$(aws lambda get-function-url-config --function-name "$FUNCTION" --region "$REGION" \
  --query FunctionUrl --output text)"

step "7/7 Cost guard: at most 2 copies of the function running at once"
if ! aws lambda put-function-concurrency --function-name "$FUNCTION" --region "$REGION" \
     --reserved-concurrent-executions 2 >/dev/null 2>&1; then
  echo "Skipped: new AWS accounts often have a concurrency quota of 10, which cannot be reserved from."
  echo "The account-wide limit still caps usage."
fi

echo
echo "Deployed! API: ${URL}"
echo "  Health:  curl ${URL}health"
echo "  Predict: curl -X POST ${URL}predict -H 'Content-Type: application/json' -d @examples/predict_request.json"
echo "  Docs:    ${URL}docs"
echo "  Logs:    aws logs tail /aws/lambda/${FUNCTION} --follow --region ${REGION}"
