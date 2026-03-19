const CONFIDENCE_THRESHOLD = 55; // percentage

function predict() {
    const file = document.getElementById("imageInput").files[0];

    if (!file) {
        alert("Please upload a cattle image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
        method: "POST",
        body: formData
    })
        .then(res => res.json())
        .then(data => {
            const resultElement = document.getElementById("result");
            const confidenceElement = document.getElementById("confidence");

            // Handle backend errors gracefully
            if (data.error || typeof data.confidence !== "number") {
                resultElement.innerText = "Unable to predict. Please provide clear cattle picture.";
                resultElement.classList.remove("text-success");
                resultElement.classList.add("text-danger");
                confidenceElement.innerText = "";
                return;
            }

            if (data.confidence < CONFIDENCE_THRESHOLD) {
                resultElement.innerText = "Unable to predict. Please provide clear cattle picture.";
                resultElement.classList.remove("text-success");
                resultElement.classList.add("text-danger");
                confidenceElement.innerText = "";
            } else {
                resultElement.innerText = "Predicted Breed: " + data.breed;
                resultElement.classList.remove("text-danger");
                resultElement.classList.add("text-success");
                confidenceElement.innerText = "Confidence: " + data.confidence.toFixed(2) + "%";
            }
        })
        .catch(() => {
            const resultElement = document.getElementById("result");
            const confidenceElement = document.getElementById("confidence");
            resultElement.innerText = "Unable to predict. Please provide clear cattle picture.";
            resultElement.classList.remove("text-success");
            resultElement.classList.add("text-danger");
            confidenceElement.innerText = "";
        });
}

// Show instructions modal on initial page load (after "login"/entry)
window.addEventListener("DOMContentLoaded", () => {
    const modalElement = document.getElementById("instructionsModal");
    if (!modalElement) return;
    const instructionsModal = new bootstrap.Modal(modalElement);
    instructionsModal.show();
});
