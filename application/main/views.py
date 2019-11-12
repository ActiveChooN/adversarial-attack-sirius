from flask import jsonify, request, render_template, Response
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from werkzeug.utils import secure_filename
from . import main


class ImageForm(FlaskForm):
    image1 = FileField("Attach first image:",
                       validators=[FileRequired("Attach file first")])
    image2 = FileField("Attach second image:",
                       validators=[FileRequired("Attach file first")])


@main.route("/", methods=["GET", "POST"])
def get_index():
    form = ImageForm(csrf_enabled=False)
    if form.validate_on_submit():
        # save images from forms to files
        image1 = form.image1.data
        filename1 = "application/static/input1." \
                    + secure_filename(image1.filename).split(".")[-1]
        image1.save(filename1)
        image2 = form.image2.data
        filename2 = "application/static/input2." \
                    + secure_filename(image1.filename).split(".")[-1]
        image2.save(filename2)
        # run inference and get prediction about both images

        return render_template("index.html", form=form, is_post=True,
                               in_filename1=filename1, in_filename2=filename2)

    return render_template("index.html", form=form)


@main.app_errorhandler(404)
def not_found_error(e):
    if request.accept_mimetypes.accept_json and \
            not request.accept_mimetypes.accept_html:
        response = jsonify({"error": "not found",
                            "message": "requested url ont found"})
        response.status_code = 404
        return response
    return render_template("404.html"), 404
