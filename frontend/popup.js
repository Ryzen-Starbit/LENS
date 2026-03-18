document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("verifyBtn").addEventListener("click", verify);
});
async function verify() {
    const text = document.getElementById("news").value;
    const response = await fetch("http://127.0.0.1:8000/verify", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
    });
    const data = await response.json();
    document.getElementById("score").innerText =
        "Truth Score: " + data.truth_percentage + "%";
    document.getElementById("output").innerText =
        JSON.stringify(data, null, 2);
}