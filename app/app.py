from flask import Flask, request, jsonify, render_template
import boto3
import os

app = Flask(__name__)

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
APP_VERSION = os.getenv("APP_VERSION", "v3")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "").strip()


@app.route("/")
def home():
    return render_template("index.html")



@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/version")
def version():
    return jsonify({"version": APP_VERSION}), 200


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "The question field is required."}), 400

    if not BEDROCK_MODEL_ID:
        return jsonify(
            {
                "error": "BEDROCK_MODEL_ID is not configured.",
                "version": APP_VERSION,
            }
        ), 500

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
        )

        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[
                {
                    "text": (
                        "You are a helpful customer-support assistant. "
                        "Give clear, concise, and safe answers."
                    )
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": [{"text": question}],
                }
            ],
            inferenceConfig={
                "maxTokens": 300,
                "temperature": 0.3,
            },
        )

        answer = response["output"]["message"]["content"][0]["text"]

        return jsonify(
            {
                "question": question,
                "answer": answer,
                "version": APP_VERSION,
            }
        ), 200

    except Exception as error:
        app.logger.exception("Amazon Bedrock invocation failed")

        return jsonify(
            {
                "error": "Amazon Bedrock invocation failed.",
                "details": str(error),
                "error_type": type(error).__name__,
                "version": APP_VERSION,
            }
        ), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
