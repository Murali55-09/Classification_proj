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
        const resultElement = document.getElementById("result");
        const confidenceElement = document.getElementById("confidence");

        if (data.confidence < 70) {
            resultElement.innerText = "Unavailable to predict, please provide clear image";
            resultElement.classList.replace("text-success", "text-danger");
            confidenceElement.innerText = "";
        } else {
            resultElement.innerText = "Predicted Breed: " + data.breed;
            resultElement.classList.replace("text-danger", "text-success");
            confidenceElement.innerText = "Confidence: " + data.confidence.toFixed(2) + "%";
        }
    })
    .catch(() => alert("Prediction failed"));
}

// Show instructions modal on load
window.addEventListener('DOMContentLoaded', () => {
    const instructionsModal = new bootstrap.Modal(document.getElementById('instructionsModal'));
    instructionsModal.show();
});
