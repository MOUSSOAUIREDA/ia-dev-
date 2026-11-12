
const API_URL = "http://localhost:8000";

const form = document.getElementById("sentiment-form");
const input = document.getElementById("review-input");
const button = document.getElementById("analyze-button");
const buttonText = document.getElementById("button-text");

const charCount = document.getElementById("char-count");
const resultSection = document.getElementById("result-section");
const errorMessage = document.getElementById("error-message");

const statusIndicator = document.getElementById("status-indicator");
const statusText = document.getElementById("status-text");

input.addEventListener("input", () => {
    charCount.textContent = `${input.value.length} / 2000`;
});

async function checkAPI() {
    try {
        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API indisponible");
        }

        const data = await response.json();

        statusIndicator.className = "status-dot online";
        statusText.textContent = `API connectée · ${data.model}`;

    } catch (error) {
        statusIndicator.className = "status-dot offline";
        statusText.textContent = "API déconnectée";
    }
}

function displayError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
    resultSection.classList.add("hidden");
}

function displayResult(data) {
    const labels = {
        POSITIVE: {
            title: "POSITIF",
            icon: "✓",
            description: "Cet avis exprime un sentiment positif."
        },
        NEGATIVE: {
            title: "NÉGATIF",
            icon: "✕",
            description: "Cet avis exprime un sentiment négatif."
        },
        INDETERMINE: {
            title: "INDÉTERMINÉ",
            icon: "?",
            description: "Le modèle ne dispose pas d'une confiance suffisante pour donner un verdict."
        }
    };

    const label = labels[data.label];

    if (!label || typeof data.score !== "number") {
        throw new Error("Format de réponse inattendu.");
    }

    resultSection.className = `result ${data.label.toLowerCase()}`;
    resultSection.classList.remove("hidden");

    document.getElementById("result-label").textContent = label.title;
    document.getElementById("result-icon").textContent = label.icon;

    const percentage = Math.round(data.score * 100);

    document.getElementById("confidence-value").textContent =
        `${percentage} %`;

    document.getElementById("confidence-bar").style.width =
        `${percentage}%`;

    document.getElementById("result-description").textContent =
        label.description;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const text = input.value.trim();

    errorMessage.classList.add("hidden");
    resultSection.classList.add("hidden");

    if (!text) {
        displayError("Veuillez saisir un avis avant de lancer l'analyse.");
        return;
    }

    button.disabled = true;
    buttonText.textContent = "Analyse en cours...";

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text })
        });

        const data = await response.json();

        if (!response.ok) {
            const detail = data.detail;

            throw new Error(
                typeof detail === "string"
                    ? detail
                    : "Une erreur est survenue pendant l'analyse."
            );
        }

        displayResult(data);

    } catch (error) {
        if (error instanceof TypeError) {
            displayError(
                "Impossible de contacter l'API. Vérifiez qu'elle est démarrée."
            );
        } else {
            displayError(error.message);
        }

    } finally {
        button.disabled = false;
        buttonText.textContent = "Analyser le sentiment";
    }
});

checkAPI();
setInterval(checkAPI, 15000);
