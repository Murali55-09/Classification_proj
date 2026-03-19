function predict() {
    let file = document.getElementById("imageInput").files[0];

    if (!file) {
        alert("Please upload an image");
        return;
    }

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerText =
            "Predicted Breed: " + data.breed;

        document.getElementById("confidence").innerText =
            "Confidence: " + data.confidence.toFixed(2) + "%";
    })
    .catch(() => alert("Prediction failed"));
}
