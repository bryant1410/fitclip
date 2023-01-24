import json
import random
from typing import Tuple

from flask import Flask, Response, jsonify, request
from flask_cors import CORS, cross_origin

from demo.search import search_in_subtitles

app = application = Flask(__name__, static_url_path="/")  # `application` is gunicorn's default, `app` is flask's.
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/search")
@cross_origin()
def search() -> Tuple[Response, int]:
    try:
        query = request.args.get("q", "[]")
        pattern = json.loads(query)  # [{"LOWER": "this", "DEP": {"IN": ["nsubj", "dobj", "iobj"]}}]
        top_k = int(request.args.get("top_k", "10"))
        results = list(search_in_subtitles(pattern))
        return jsonify([
            {
                "video_id": span.doc._.video_id,
                "start_time": span[0]._.start_time,
                "end_time": span[-1]._.end_time,
                "text": span.text,
            }
            for span in random.sample(results, min(top_k, len(results)))
        ]), 200
    except Exception as e:  # noqa
        return jsonify(status=500, message=repr(e)), 500


@app.route("/")
def root() -> Response:
    return app.send_static_file("index.html")
