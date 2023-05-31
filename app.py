from flask import Flask, jsonify, request

# Different functions for different plant types are imported here from plant folder

from guava.guava import guava

#strting of an flask application

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def predict():

    request_data = request.get_json()

    imageString = request_data['img']

    prediction = ''

    prediction = guava(imageString)


    print(prediction)

    return prediction


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)