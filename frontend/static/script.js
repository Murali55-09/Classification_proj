const CONFIDENCE_THRESHOLD = 40; // percentage

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

            console.log("Prediction response:", data);

            // If backend sent an explicit error, show it
            if (data.error) {
                resultElement.innerText = "Error: " + data.error;
                resultElement.classList.remove("text-success");
                resultElement.classList.add("text-danger");
                confidenceElement.innerText = "";
                return;
            }

            // Try to interpret confidence as a number even if it comes as string
            const confidenceValue = Number(data.confidence);

            if (Number.isNaN(confidenceValue)) {
                resultElement.innerText = "Unable to predict. Please provide clear cattle picture.";
                resultElement.classList.remove("text-success");
                resultElement.classList.add("text-danger");
                confidenceElement.innerText = "";
                return;
            }

            if (confidenceValue < CONFIDENCE_THRESHOLD) {
                resultElement.innerText = "Unable to predict. Please provide clear cattle picture.";
                resultElement.classList.remove("text-success");
                resultElement.classList.add("text-danger");
                // Still show the raw confidence so you can debug
                confidenceElement.innerText = "Model confidence: " + confidenceValue.toFixed(2) + "% (below " + CONFIDENCE_THRESHOLD + "% threshold)";
            } else {
                resultElement.innerText = "Predicted Breed: " + data.breed;
                resultElement.classList.remove("text-danger");
                resultElement.classList.add("text-success");
                confidenceElement.innerText = "Confidence: " + confidenceValue.toFixed(2) + "%";
            }
        })
        .catch((err) => {
            console.error("Prediction error:", err);
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
