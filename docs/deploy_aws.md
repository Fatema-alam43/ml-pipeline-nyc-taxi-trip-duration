# Deploying the API to AWS Lambda

The API runs as a **container image on AWS Lambda** with a public HTTPS **function URL**. Lambda only runs (and
bills) while a request is being handled, so a portfolio demo costs close to nothing.
The same image also runs locally with Docker and on ECS/Fargate (see the end of this page).

```
models/lgbm_improved.txt ──► S3 bucket ──(downloaded at startup)──┐
Dockerfile ──► image ──► ECR repository ──► Lambda function ◄─────┘ ──► https://<id>.lambda-url.<region>.on.aws/
```

## 0. One-time setup

1. **Budget alert first.** AWS console → **Billing and Cost Management → Budgets → Create budget**
   → *Use a template* → **Monthly cost budget** → amount **$5** → your email → **Create budget**.
2. **AWS CLI v2** (recent, for the function URL permission flags): `brew install awscli` or `brew upgrade awscli`,
   then `aws --version` (2.31 or newer).
3. **Credentials:** `aws configure` with an IAM user's access key (not the root account), region `ca-central-1`.
   Check with `aws sts get-caller-identity`.
4. **Docker Desktop** running: `docker info`.

## 1. Deploy

```bash
python scripts/pin_api_requirements.py     # API image uses the same package versions as training
git add -A && git commit -m "..."          # the image is tagged with the git commit
bash deploy/deploy_lambda.sh
```

The script (safe to re-run) uploads the model to S3, builds an arm64 image and pushes it to ECR, creates an IAM role
that can only read the model, creates or updates the Lambda function, and prints the public URL.

## 2. Try it

```bash
URL=https://<your-id>.lambda-url.ca-central-1.on.aws/
curl ${URL}health
curl -X POST ${URL}predict -H "Content-Type: application/json" -d @examples/predict_request.json
open ${URL}docs                                                   # interactive API docs
aws logs tail /aws/lambda/nyc-taxi-api --follow --region ca-central-1   # live logs
```

The first request after a quiet period takes a few seconds (a *cold start*: Lambda starts the container and
downloads the model). Later requests take milliseconds.

## 3. Shut it down

```bash
bash deploy/teardown_lambda.sh
```

Deletes the function and its URL, the log group, the image repository, the model bucket and the IAM role.

## Costs

| Resource | Cost for a demo |
|---|---|
| Lambda (2 GB, arm64) | Free tier: 1 million requests and 400,000 GB-seconds per month |
| ECR (~0.5 GB image) | About $0.05 per month |
| S3 (~8 MB model) | Under $0.01 per month |
| CloudWatch logs | Free tier: 5 GB per month |

Concurrency is capped where the account allows it, and the budget alert emails you if spending passes $5.

## Alternative: ECS Express Mode

AWS App Runner is closed to new customers (since April 30, 2026); AWS recommends **ECS Express Mode**, which runs
the same image on Fargate behind a load balancer with an HTTPS URL. Use the image already in ECR, container port
**8080**, health check path **/health**, and give the task role `s3:GetObject` on the model. The load balancer is
billed for every hour it exists, so delete the service after the demo.
