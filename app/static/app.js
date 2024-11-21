const outputDiv = document.getElementById("output");

// Handle form submissions
async function handleFormSubmission(formId, endpoint, formData = null, method = "POST") {
    try {
        const response = await fetch(endpoint, {
            method: method,
            body: formData || undefined,
        });
        const result = await response.json();
        outputDiv.innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
    } catch (error) {
        outputDiv.innerHTML = `<p>Error: ${error.message}</p>`;
    }
}

// Train the Model
document.getElementById("trainForm").addEventListener("submit", (e) => {
    e.preventDefault();
    handleFormSubmission("trainForm", "/train");
});

// Make Predictions
document.getElementById("predictForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const inputData = document.getElementById("inputData").value;
    handleFormSubmission("predictForm", "/predict", JSON.stringify(JSON.parse(inputData)), "POST");
});

// Calculate RFMS
document.getElementById("rfmsForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("rfmsFile").files[0];
    const formData = new FormData();
    formData.append("file", fileInput);
    handleFormSubmission("rfmsForm", "/rfms", formData);
});

// Classify Customers
document.getElementById("goodBadForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("goodBadFile").files[0];
    const formData = new FormData();
    formData.append("file", fileInput);
    handleFormSubmission("goodBadForm", "/good_bad", formData);
});

// Calculate WoE
document.getElementById("woeForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("woeFile").files[0];
    const formData = new FormData();
    formData.append("file", fileInput);
    formData.append("target", document.getElementById("targetColumn").value);
    formData.append("features", document.getElementById("featureColumns").value.split(","));
    handleFormSubmission("woeForm", "/woe", formData);
});
