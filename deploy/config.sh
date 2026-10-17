# Shared settings for the deploy and teardown scripts. Change REGION here if you like.
export AWS_PAGER=""                                   # no interactive pager for AWS CLI output
REGION="${AWS_REGION:-ca-central-1}"                  # Canada (Central)
APP="nyc-taxi-api"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
BUCKET="${APP}-models-${ACCOUNT_ID}-${REGION}"        # bucket names must be globally unique
REPO="${APP}"
ROLE="${APP}-lambda-role"
FUNCTION="${APP}"
ECR="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
