#!/usr/bin/env bash
# Delete everything deploy_lambda.sh created, so nothing keeps costing money.
#   bash deploy/teardown_lambda.sh
set -uo pipefail          # no -e: keep going if something was already deleted
cd "$(dirname "$0")/.."
source deploy/config.sh

read -r -p "Delete the Lambda function, image repository, model bucket and role in ${REGION}? [y/N] " answer
[ "$answer" = "y" ] || { echo "Cancelled."; exit 0; }

echo "==> Lambda function (its URL is deleted with it)"
aws lambda delete-function --function-name "$FUNCTION" --region "$REGION"
echo "==> Log group"
aws logs delete-log-group --log-group-name "/aws/lambda/${FUNCTION}" --region "$REGION"
echo "==> ECR repository and images"
aws ecr delete-repository --repository-name "$REPO" --force --region "$REGION" >/dev/null
echo "==> S3 bucket and model files"
aws s3 rb "s3://${BUCKET}" --force --region "$REGION"
echo "==> IAM role"
aws iam delete-role-policy --role-name "$ROLE" --policy-name read-model
aws iam detach-role-policy --role-name "$ROLE" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam delete-role --role-name "$ROLE"
echo "Done. Check the AWS Billing console over the next day to confirm nothing is left."
