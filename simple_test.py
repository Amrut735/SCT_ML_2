from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, Flask is working!"

@app.route('/test')
def test():
    return "Test route is working!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 