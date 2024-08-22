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

# Store generated questions in memory
asked_questions = []


def create_prompt(text_input):
    return (
        f"Please provide a detailed explanation about the following topic: {text_input}"
    )


def extract_key_points(text):
    key_points = text.split(". ")
    return [point.strip() for point in key_points if point.strip()]


def generate_quiz(text):
    key_points = extract_key_points(text)
    questions = []

    for point in key_points:
        if point:
            question = f"What is the main idea of the following statement: '{point}'?"
            correct_option = f"Option A: {point}"
            incorrect_options = generate_incorrect_options(point, key_points)

            options = [correct_option] + incorrect_options
            random.shuffle(options)

            questions.append(
                {"question": question, "options": options, "answer": "Option A"}
            )

    return questions


def generate_incorrect_options(correct_point, key_points):
    incorrect_options = []
    while len(incorrect_options) < 3:
        random_point = random.choice(key_points)
        if random_point != correct_point and random_point not in incorrect_options:
            incorrect_options.append(f"Option B: {random_point}")

    while len(incorrect_options) < 3:
        incorrect_options.append(f"Option B: Incorrect Answer")

    return incorrect_options


@app.route("/ask_question", methods=["POST"])
def ask_question():
    data = request.json
    text_input = data.get("text")

    if not text_input:
        return jsonify({"error": "Text input is required"}), 400

    prompt = create_prompt(text_input)
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
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
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
        content = generated_text.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            response_text = content[0].get("text", "")
        else:
            response_text = ""

        # Generate speech
        audio_file = synthesize_speech(response_text)

        # Generate quiz based on the response
        quiz = generate_quiz(response_text)

        # Store generated questions for future reference
        asked_questions.extend(quiz)

        return jsonify(
            {"generated_text": response_text, "audio_file": audio_file, "quiz": quiz}
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/submit_answer", methods=["POST"])
def submit_answer():
    data = request.json
    question = data.get("question")
    user_answer = data.get("answer")

    if not question or not user_answer:
        return jsonify({"error": "Question and answer are required"}), 400

    for quiz in asked_questions:
        if quiz["question"] == question:
            if quiz["answer"] == user_answer:
                return jsonify({"result": "Correct!"})
            else:
                return jsonify(
                    {"result": "Incorrect. The correct answer was: " + quiz["answer"]}
                )

    return jsonify({"error": "Question not found"}), 404


def synthesize_speech(generated_text):
    try:
        polly_response = polly.synthesize_speech(
            Text=generated_text,
            OutputFormat="mp3",
            VoiceId="Joanna",
        )

        audio_file = "output.mp3"
        with open(audio_file, "wb") as file:
            file.write(polly_response["AudioStream"].read())

        return audio_file

    except Exception as e:
        raise Exception(f"Error synthesizing speech: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
