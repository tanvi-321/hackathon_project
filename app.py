from flask import Flask, request, jsonify
from dotenv import load_dotenv
import boto3
import json
import random
import os

load_dotenv()

app = Flask(__name__)
REGION = os.environ.get("aws_region")
ACCESS_KEY = os.environ.get("aws_access_key")
SECRET_KEY = os.environ.get("aws_secret_key")


brt = boto3.client(
    # "kendra",
    service_name="bedrock-runtime",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION,
)
polly = boto3.client(
    service_name="polly",
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION,
)


def create_prompt(text_input):
    # Customize the prompt based on the input or requirements
    return (
        f"Please provide a detailed explanation about the following topic: {text_input}"
    )


@app.route("/generate", methods=["POST"])
def generate_response():
    data = request.json
    text_input = data.get("text")

    if not text_input:
        return jsonify({"error": "Text input is required"}), 400

    prompt = create_prompt(text_input)
    # Prepare the request body
    request_body = {
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3000,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
        },
    }
    request_body = {
        "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "contentType": "application/json",
        "accept": "application/json",
        "body": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 3000,
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 250,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": text_input}]},
            ],
        },
    }

    try:
        response = brt.invoke_model(
            modelId=request_body["modelId"],
            contentType=request_body["contentType"],
            accept=request_body["accept"],
            body=json.dumps(request_body["body"]),
        )

        generated_text = json.loads(response["body"].read().decode("utf-8"))
        # return jsonify(generated_text)
        # return generated_text

        # Extract the text from the content
        content = generated_text.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            generated_text = content[0].get("text", "")
        else:
            generated_text = ""

        return generated_text

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def synthesize_speech(generated_text):
    try:
        polly_response = polly.synthesize_speech(
            Text=generated_text,
            OutputFormat="mp3",
            VoiceId="Joanna",  # Change to desired voice
        )

        audio_file = "output.mp3"
        with open(audio_file, "wb") as file:
            file.write(polly_response["AudioStream"].read())

        return audio_file

    except Exception as e:
        raise Exception(f"Error synthesizing speech: {str(e)}")


@app.route("/speech_generate", methods=["POST"])
def generate_response_api():
    # data = request.json
    # text_input = data.get("text")

    # if not text_input:
    #     return jsonify({"error": "Text input is required"}), 400

    try:
        generated_text = generate_response()
        print(generated_text)
        audio_file = synthesize_speech(generated_text)

        return jsonify({"generated_text": generated_text, "audio_file": audio_file})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
