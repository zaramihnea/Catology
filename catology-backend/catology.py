from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    print(f"Received message: {user_message}")

    response = "I am alive"
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)